import argparse
import joblib
import pandas as pd
import numpy as np
import itertools
import os
import sys

# Define amino acid alphabet and dipeptides
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
DIPEPTIDES = [''.join(p) for p in itertools.product(AA_ALPHABET, repeat=2)]

def calc_aac(sequence):
    length = len(sequence)
    if length == 0:
        return {aa: 0 for aa in AA_ALPHABET}
    counts = {aa: sequence.count(aa) for aa in AA_ALPHABET}
    return {aa: count / length for aa, count in counts.items()}

def calc_dpc(sequence):
    length = len(sequence)
    dpc_counts = {dp: 0 for dp in DIPEPTIDES}
    if length < 2:
        return dpc_counts
    for i in range(length - 1):
        dp = sequence[i:i+2]
        if dp in dpc_counts:
            dpc_counts[dp] += 1
    total_dp = length - 1
    return {dp: count / total_dp for dp, count in dpc_counts.items()}

def extract_features(sequence):
    sequence = str(sequence).upper().strip()
    aac = calc_aac(sequence)
    dpc = calc_dpc(sequence)
    
    aac_df = pd.DataFrame([aac])
    dpc_df = pd.DataFrame([dpc])
    feat_df = pd.concat([aac_df, dpc_df], axis=1)
    
    return feat_df.values

def load_models(model_dir):
    model_path = os.path.join(model_dir, 'best_ensemble_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
        print(f"Error: Model files not found in '{model_dir}'. Make sure best_classical_model.pkl, scaler.pkl, and label_encoder.pkl exist.")
        sys.exit(1)
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    return model, scaler, le

def predict_sequence(sequence, model, scaler, le):
    features = extract_features(sequence)
    features_scaled = scaler.transform(features)
    probabilities = model.predict_proba(features_scaled)[0]
    
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_classes = le.inverse_transform(top_3_indices)
    top_3_probs = probabilities[top_3_indices]
    
    return list(zip(top_3_classes, top_3_probs))

def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        name = ""
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    sequences.append((name, seq))
                name = line[1:]
                seq = ""
            else:
                seq += line
        if name:
            sequences.append((name, seq))
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Multi-class Peptide Family Predictor CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--sequence', type=str, help='A single peptide sequence to predict')
    group.add_argument('-f', '--fasta', type=str, help='Path to a FASTA file containing multiple sequences')
    
    parser.add_argument('-m', '--model-dir', type=str, default='models', help='Path to the directory containing model .pkl files (default: "models")')
    parser.add_argument('-o', '--output', type=str, help='Path to save the output as a CSV file (optional)')
    
    args = parser.parse_args()
    
    model, scaler, le = load_models(args.model_dir)
    
    results = []
    
    if args.sequence:
        seq = args.sequence.upper().strip()
        preds = predict_sequence(seq, model, scaler, le)
        print(f"\nSequence: {seq}")
        print(f"{'Class':<20} {'Prediction Score'}")
        print("-" * 40)
        for cls, prob in preds:
            print(f"{cls:<20} {prob:.3f}")
        
        # Prepare for CSV if requested
        row = {'Sequence_ID': 'Input_Sequence', 'Sequence': seq}
        for i, (cls, prob) in enumerate(preds):
            row[f'Top_{i+1}_Class'] = cls
            row[f'Top_{i+1}_Score'] = prob
        results.append(row)
            
    elif args.fasta:
        print(f"Reading FASTA file: {args.fasta}...")
        sequences = parse_fasta(args.fasta)
        print(f"Found {len(sequences)} sequences.\n")
        
        for name, seq in sequences:
            preds = predict_sequence(seq, model, scaler, le)
            print(f">{name}")
            for cls, prob in preds:
                print(f"  {cls:<20} : {prob:.3f}")
            print()
            
            row = {'Sequence_ID': name, 'Sequence': seq}
            for i, (cls, prob) in enumerate(preds):
                row[f'Top_{i+1}_Class'] = cls
                row[f'Top_{i+1}_Score'] = round(prob, 3)
            results.append(row)
            
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults successfully saved to {args.output}")

if __name__ == "__main__":
    main()

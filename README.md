# Multi-class Peptide Family Predictor

This repository contains a standalone command-line tool (`predict.py`) to classify peptide sequences into one of 9 distinct families (e.g., CLE, Cyclotides, Defensins, LTPs, PSKs, RALFs, Snakins, Thionins, Other). 

The predictor uses a trained LightGBM machine learning model based on Amino Acid Composition (AAC) and Dipeptide Composition (DPC) features.

## Prerequisites

Make sure you have Python 3 installed along with the required libraries.

```bash
pip install pandas numpy scikit-learn lightgbm joblib
```

*Note: The model files must exist in the `models/` directory (`best_classical_model.pkl`, `scaler.pkl`, `label_encoder.pkl`).*

## How to Run

You can run the predictor using the `predict.py` script. It accepts either a single sequence or a FASTA file containing multiple sequences.

### 1. Predict a Single Sequence

Use the `-s` or `--sequence` flag to pass a single peptide string.

```bash
python predict.py -s "MAQSLTLIFVILILGLASLASSARAEKQLAEKAAAKLAEKAAAKLAEKAA"
```

**Output:**
```
Sequence: MAQSLTLIFVILILGLASLASSARAEKQLAEKAAAKLAEKAAAKLAEKAA
Class                Prediction Score
----------------------------------------
CLE                  0.902
LTPs                 0.051
Thionins             0.012
```

### 2. Predict from a FASTA File

Use the `-f` or `--fasta` flag to process multiple sequences from a FASTA file.

```bash
python predict.py -f dataset/temp_Thionins.fasta
```

### 3. Save Results to a CSV File

You can use the `-o` or `--output` flag to save the top 3 predictions for your sequence(s) directly to a CSV file.

```bash
python predict.py -s "MAQSLTLIFVILILGLASLASSARAEKQLAEKAAAKLAEKAAAKLAEKAA" -o results.csv
```

### Help Command

To see all available options, run:

```bash
python predict.py --help
```

# Plant Peptide Classification

A comprehensive categorization of small plant peptides based on their primary biological functions: signaling, defense, and transport.

## 1. Signaling Peptides
These peptides act as chemical messengers. They bind to specific cell-surface receptors to regulate plant growth, cell division, development, and stress responses.

*   **CLE:** CLAVATA3/EMBRYO SURROUNDING REGION-related
*   **PSKs:** Phytosulfokines
*   **RALFs:** Rapid Alkalinization Factors

## 2. Defense Peptides
These peptides are vital components of the plant's innate immune system. They directly target, inhibit, or destroy invading pathogens such as fungi, bacteria, and insects.

*   **Cyclotides:** Cyclic Peptides *(named for their unique circular backbone structure)*
*   **Defensins:** Plant Defensins
*   **Snakins:** Snakin Peptides *(named due to structural similarities to snake venom toxins)*
*   **Thionins:** Plant Thionins

## 3. Transport & Defense Peptides
These versatile proteins serve a dual purpose. They manage the transport of essential lipids and structural molecules while simultaneously providing structural defense and antimicrobial protection.

*   **LTPs:** Lipid Transfer Proteins


## Developers

**Md Faiyaz Rizwee**
📧 [mdfaiyaz4840@gmail.com](mailto:mdfaiyaz4840@gmail.com) &nbsp;|&nbsp; 🔗 [LinkedIn](https://www.linkedin.com/in/md-faiyaz-rizwee-62024438b)

**A.T. Vivek**
📧 [vivek37373@nipgr.ac.in](mailto:vivek37373@nipgr.ac.in) &nbsp;|&nbsp; 🔗 [LinkedIn](https://www.linkedin.com/in/vivek-thiruvettai/)



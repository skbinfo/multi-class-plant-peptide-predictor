import pandas as pd
import numpy as np
import logging
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/03c_ensemble_ml.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def plot_confusion_matrix(y_true, y_pred, classes, model_name, cm_matrix=None):
    if cm_matrix is None:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = cm_matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (CV Sum) - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/cm_{model_name.replace(" ", "_")}.png')
    plt.close()

def main():
    logging.info("Starting Ensemble Training...")
    
    # Load hyperparams
    with open('results/best_hyperparameters.json', 'r') as f:
        best_params = json.load(f)
        
    lgb_params = {k: v for k, v in best_params['LightGBM'].items() if k != 'best_f1_macro'}
    xgb_params = {k: v for k, v in best_params['XGBoost'].items() if k != 'best_f1_macro'}
    mlp_params = {k: v for k, v in best_params['MLP'].items() if k != 'best_f1_macro' and not k.startswith('n_units') and k != 'n_layers'}
    
    # Reconstruct MLP layers tuple
    n_layers = best_params['MLP']['n_layers']
    layers = tuple([best_params['MLP'][f'n_units_l{i}'] for i in range(n_layers)])
    mlp_params['hidden_layer_sizes'] = layers
    mlp_params['max_iter'] = 500
    mlp_params['early_stopping'] = True
    mlp_params['random_state'] = 42

    lgb_params['random_state'] = 42
    lgb_params['class_weight'] = 'balanced'
    lgb_params['n_jobs'] = -1
    lgb_params['verbose'] = -1

    xgb_params['random_state'] = 42
    xgb_params['eval_metric'] = 'mlogloss'
    xgb_params['use_label_encoder'] = False
    xgb_params['n_jobs'] = -1
    
    # Models
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_mlp = MLPClassifier(**mlp_params)
    
    # Add a Random Forest for robust diversity
    model_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[
            ('LightGBM_Tuned', model_lgb),
            ('XGBoost_Tuned', model_xgb),
            ('MLP_Tuned', model_mlp),
            ('RandomForest', model_rf)
        ],
        voting='soft'
    )
    
    # Data Loading
    train_df = pd.read_csv('features/train_features.csv')
    test_df = pd.read_csv('features/test_features.csv')
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.fillna(0)
    
    X = full_df.drop(columns=['Entry', 'Label']).values
    y_raw = full_df['Label'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    metrics = defaultdict(list)
    cm_total = np.zeros((len(classes), len(classes)), dtype=int)
    
    all_y_true = []
    all_y_pred = []
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        logging.info(f"\\n--- Starting Fold {fold}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        target_count = 400
        try:
            cyclotides_idx = list(classes).index('Cyclotides')
            smote = SMOTE(sampling_strategy={cyclotides_idx: target_count}, random_state=42)
            rus = RandomUnderSampler(sampling_strategy={i: target_count for i in range(len(classes))}, random_state=42)
            pipeline = Pipeline(steps=[('smote', smote), ('rus', rus)])
            X_train, y_train = pipeline.fit_resample(X_train, y_train)
        except ValueError:
            pass
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Since data is perfectly balanced by SMOTE/RUS, sample_weights are inherently uniform
        ensemble.fit(X_train, y_train)
        
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        
        cm_fold = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
        FP = cm_fold.sum(axis=0) - np.diag(cm_fold)
        FN = cm_fold.sum(axis=1) - np.diag(cm_fold)
        TP = np.diag(cm_fold)
        TN = cm_fold.sum() - (FP + FN + TP)
        
        metrics['Accuracy'].append(acc)
        metrics['Precision'].append(prec)
        metrics['Recall'].append(rec)
        metrics['F1_Macro'].append(f1_macro)
        metrics['F1_Weighted'].append(f1_weighted)
        metrics['MCC'].append(mcc)
        metrics['ROC_AUC'].append(roc_auc)
        metrics['TP_Macro'].append(np.mean(TP))
        metrics['TN_Macro'].append(np.mean(TN))
        metrics['FP_Macro'].append(np.mean(FP))
        metrics['FN_Macro'].append(np.mean(FN))
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        cm_total += cm_fold
        fold += 1
        
    logging.info("\\n--- ENSEMBLE CV COMPLETE ---")
    res_dict = {'Model': 'Tuned_Ensemble_Voting'}
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        res_dict[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        
    results_df = pd.DataFrame([res_dict])
    
    if os.path.exists('results/classical_ml_cv_results.csv'):
        old_df = pd.read_csv('results/classical_ml_cv_results.csv')
        # Insert ensemble at the top
        results_df = pd.concat([results_df, old_df], ignore_index=True)
        
    results_df.to_csv('results/classical_ml_cv_results.csv', index=False)
    logging.info(f"\n{results_df.head(3).to_string()}")
    
    # Save Classification Report
    report_str = classification_report(all_y_true, all_y_pred, target_names=classes)
    with open('results/best_model_classification_report.txt', 'w') as f:
        f.write("Best Model: Tuned Ensemble Voting (5-Fold CV aggregated)\n\n")
        f.write(report_str)
    logging.info("Classification report saved to results/best_model_classification_report.txt")
    
    plot_confusion_matrix(None, None, classes, 'Tuned_Ensemble_Voting', cm_matrix=cm_total)
    joblib.dump(ensemble, 'models/best_ensemble_model.pkl')
    logging.info("Ensemble model saved.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/03_train_ml.log', filemode='w')
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

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] # top 20
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.bar(range(20), importances[indices], align='center')
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, 20])
        plt.tight_layout()
        plt.savefig(f'plots/feat_imp_{model_name.replace(" ", "_")}.png')
        plt.close()

def main():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logging.info("Loading features...")
    train_df = pd.read_csv('features/train_features.csv')
    test_df = pd.read_csv('features/test_features.csv')
    
    # Combine train and test to use for CV
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.fillna(0)
    
    X = full_df.drop(columns=['Entry', 'Label']).values
    y_raw = full_df['Label'].values
    feature_names = full_df.drop(columns=['Entry', 'Label']).columns
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    joblib.dump(le, 'models/label_encoder.pkl')
    
    classes = le.classes_
    logging.info(f"Classes: {classes}")
    
    # Initialize CV
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    metrics_per_model = defaultdict(lambda: defaultdict(list))
    cms = defaultdict(lambda: np.zeros((len(classes), len(classes)), dtype=int))
    
    global_best_f1 = -1
    global_best_model_name = ""
    global_best_model = None
    global_best_scaler = None
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        logging.info(f"\\n--- Starting Fold {fold}/{n_splits} ---")
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Balance classes: SMOTE for Cyclotides, undersample others to 400
        target_count = 400
        try:
            cyclotides_idx = list(classes).index('Cyclotides')
            smote_strategy = {cyclotides_idx: target_count}
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
            
            rus_strategy = {i: target_count for i in range(len(classes))}
            rus = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=42)
            
            pipeline = Pipeline(steps=[('smote', smote), ('rus', rus)])
            X_train_fold, y_train_fold = pipeline.fit_resample(X_train_fold, y_train_fold)
        except ValueError:
            pass # Keep quiet during CV
            
        # Scale
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_fold)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True), 
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1),
            'CatBoost': CatBoostClassifier(iterations=200, random_seed=42, verbose=0)
        }
        
        for name, model in models.items():
            try:
                if name in ['XGBoost', 'CatBoost']:
                    model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights)
                elif name == 'MLP':
                    model.fit(X_train_fold, y_train_fold)
                else:
                    model.fit(X_train_fold, y_train_fold)
                    
                y_pred = model.predict(X_test_fold)
                y_proba = model.predict_proba(X_test_fold)
                
                acc = accuracy_score(y_test_fold, y_pred)
                prec = precision_score(y_test_fold, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test_fold, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test_fold, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_test_fold, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_test_fold, y_pred)
                roc_auc = roc_auc_score(y_test_fold, y_proba, multi_class='ovr', average='macro')
                
                cm_fold = confusion_matrix(y_test_fold, y_pred, labels=range(len(classes)))
                FP = cm_fold.sum(axis=0) - np.diag(cm_fold)
                FN = cm_fold.sum(axis=1) - np.diag(cm_fold)
                TP = np.diag(cm_fold)
                TN = cm_fold.sum() - (FP + FN + TP)
                
                metrics_per_model[name]['Accuracy'].append(acc)
                metrics_per_model[name]['Precision'].append(prec)
                metrics_per_model[name]['Recall'].append(rec)
                metrics_per_model[name]['F1_Macro'].append(f1_macro)
                metrics_per_model[name]['F1_Weighted'].append(f1_weighted)
                metrics_per_model[name]['MCC'].append(mcc)
                metrics_per_model[name]['ROC_AUC'].append(roc_auc)
                metrics_per_model[name]['TP_Macro'].append(np.mean(TP))
                metrics_per_model[name]['TN_Macro'].append(np.mean(TN))
                metrics_per_model[name]['FP_Macro'].append(np.mean(FP))
                metrics_per_model[name]['FN_Macro'].append(np.mean(FN))
                
                cms[name] += cm_fold
                
                if f1_macro > global_best_f1:
                    global_best_f1 = f1_macro
                    global_best_model_name = name
                    global_best_model = model
                    global_best_scaler = scaler
                    
            except Exception as e:
                logging.error(f"Failed to train {name} on Fold {fold}: {e}")
                
        fold += 1
        
    logging.info("\\n--- CV COMPLETE. AGGREGATING RESULTS ---")
    results = []
    for name, metrics in metrics_per_model.items():
        res_dict = {'Model': name}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            res_dict[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
            res_dict[f"{metric_name}_mean"] = mean_val # for sorting
        results.append(res_dict)
        
        plot_confusion_matrix(None, None, classes, name, cm_matrix=cms[name])
        plot_feature_importance(models[name], feature_names, name)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1_Macro_mean', ascending=False)
    display_df = results_df.drop(columns=[c for c in results_df.columns if c.endswith('_mean')])
    display_df.to_csv('results/classical_ml_cv_results.csv', index=False)
    
    # Print clean summary table in log
    logging.info(f"\\n{display_df.to_string()}")
    
    logging.info(f"Absolute Best Model Single-Fold Run: {global_best_model_name} with F1_Macro: {global_best_f1:.4f}")
    joblib.dump(global_best_model, 'models/best_classical_model.pkl')
    joblib.dump(global_best_scaler, 'models/scaler.pkl')

if __name__ == "__main__":
    main()

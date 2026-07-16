import optuna
import pandas as pd
import numpy as np
import logging
import json
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/03b_tune_ml.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Suppress optuna spam in console, keep in log
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, model_name, X, y, classes):
    n_splits = 3 # Use 3-fold for faster tuning
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
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
        X_val = scaler.transform(X_val)
        
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        if model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1,
                'verbose': -1
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'n_jobs': -1
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
        elif model_name == 'MLP':
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layers = []
            for i in range(n_layers):
                layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256, step=32))
                
            params = {
                'hidden_layer_sizes': tuple(layers),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': 'adam', # Explicitly using ADAM solver
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': 500,
                'early_stopping': True,
                'random_state': 42
            }
            model = MLPClassifier(**params)
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)
        
    return np.mean(f1_scores)

def main():
    logging.info("Starting Hyperparameter Tuning...")
    train_df = pd.read_csv('features/train_features.csv')
    test_df = pd.read_csv('features/test_features.csv')
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.fillna(0)
    
    X = full_df.drop(columns=['Entry', 'Label']).values
    y_raw = full_df['Label'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    
    best_params_all = {}
    
    # 30 trials per model to find optimal settings quickly
    n_trials = 30
    
    for model_name in ['LightGBM', 'XGBoost', 'MLP']:
        logging.info(f"\\n================================")
        logging.info(f"Tuning {model_name}...")
        study = optuna.create_study(direction='maximize', study_name=model_name)
        study.optimize(lambda trial: objective(trial, model_name, X, y, classes), n_trials=n_trials)
        
        logging.info(f"Best {model_name} params: {study.best_params}")
        logging.info(f"Best {model_name} F1_Macro: {study.best_value:.4f}")
        
        best_params_all[model_name] = study.best_params
        best_params_all[model_name]['best_f1_macro'] = study.best_value
        
    with open('results/best_hyperparameters.json', 'w') as f:
        json.dump(best_params_all, f, indent=4)
    logging.info("\\nHyperparameter tuning complete. Saved to results/best_hyperparameters.json")

if __name__ == "__main__":
    main()

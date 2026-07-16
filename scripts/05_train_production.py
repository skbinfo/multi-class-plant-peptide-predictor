import pandas as pd
import numpy as np
import logging
import json
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/05_train_production.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def main():
    logging.info("Starting Production Model Training on 100% of data...")
    
    # 1. Load Data
    train_df = pd.read_csv('features/train_features.csv')
    test_df = pd.read_csv('features/test_features.csv')
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.fillna(0)
    
    X = full_df.drop(columns=['Entry', 'Label']).values
    y_raw = full_df['Label'].values
    
    # 2. Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    joblib.dump(le, 'models/label_encoder.pkl')
    logging.info("LabelEncoder saved.")
    
    # 3. Apply SMOTE + RUS to the entire dataset
    target_count = 400
    try:
        cyclotides_idx = list(classes).index('Cyclotides')
        smote = SMOTE(sampling_strategy={cyclotides_idx: target_count}, random_state=42)
        rus = RandomUnderSampler(sampling_strategy={i: target_count for i in range(len(classes))}, random_state=42)
        pipeline = Pipeline(steps=[('smote', smote), ('rus', rus)])
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        logging.info("Applied SMOTE and RUS to 100% of data.")
    except ValueError:
        X_resampled, y_resampled = X, y
        logging.warning("Could not apply SMOTE/RUS. Proceeding with original distribution.")

    # 4. Fit Global Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    joblib.dump(scaler, 'models/scaler.pkl')
    logging.info("Global StandardScaler saved to models/scaler.pkl")

    # 5. Load Hyperparameters
    with open('results/best_hyperparameters.json', 'r') as f:
        best_params = json.load(f)
        
    lgb_params = {k: v for k, v in best_params['LightGBM'].items() if k != 'best_f1_macro'}
    xgb_params = {k: v for k, v in best_params['XGBoost'].items() if k != 'best_f1_macro'}
    mlp_params = {k: v for k, v in best_params['MLP'].items() if k != 'best_f1_macro' and not k.startswith('n_units') and k != 'n_layers'}
    
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
    
    # 6. Initialize Models
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_mlp = MLPClassifier(**mlp_params)
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
    
    # 7. Train Master Model
    logging.info("Training master ensemble model on perfectly balanced 100% dataset...")
    ensemble.fit(X_scaled, y_resampled)
    
    joblib.dump(ensemble, 'models/best_ensemble_model.pkl')
    logging.info("Master model successfully saved to models/best_ensemble_model.pkl")
    logging.info("Production training is COMPLETE! The pipeline is ready.")

if __name__ == "__main__":
    main()

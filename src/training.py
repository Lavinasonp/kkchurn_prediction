import os
import yaml
import joblib
import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# Import our custom Transformer from the file we just created
from src.transformers import KKBoxFeatureEngineering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(config_path="config/config.yaml", params_path="config/params.yaml"):
    # 1. Load Configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    # 2. Load Gold Data
    final_data_path = os.path.join(config['directories']['final_data'], config['files']['final_output'])
    logging.info(f"⏳ Loading training data from {final_data_path}...")
    
    if not os.path.exists(final_data_path):
        logging.error("❌ Data not found! You must run the feature engineering step first.")
        return

    df = pd.read_parquet(final_data_path)

    # 3. Prepare X and y
    target = params['training']['target_col']
    unused_cols = params['training']['unused_cols']
    
    # Drop target and strictly unused columns (ID columns, dates)
    X = df.drop(columns=[target] + unused_cols, errors='ignore')
    y = df[target]

    logging.info(f"📊 Input Feature Shape: {X.shape}")

    # 4. Define the Pipeline
    # Step A: Custom Feature Engineering (Median Imputation, Ratios, Casting)
    # Step B: LightGBM Classifier
    pipeline = Pipeline([
        ('feature_eng', KKBoxFeatureEngineering()),
        ('model', LGBMClassifier(**params['lgbm_params']))
    ])

    # 5. Setup Local MLflow
    # This will create an 'mlruns' folder in your project root
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "KKBox_Churn_Local"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        logging.info(f"🚀 Started MLflow Run ID: {run.info.run_id}")
        
        # Log params
        mlflow.log_params(params['lgbm_params'])
        mlflow.log_param("input_rows", X.shape[0])

        # --- Phase 1: Cross-Validation ---
        logging.info("⚔️  Starting Stratified Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=params['training']['n_splits'], 
                              shuffle=True, 
                              random_state=params['training']['random_state'])
        
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Clone pipeline for fresh start
            from sklearn.base import clone
            fold_pipe = clone(pipeline)
            
            fold_pipe.fit(X_train, y_train)
            
            # Predict
            preds = fold_pipe.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
            auc_scores.append(score)
            
            logging.info(f"   Fold {fold+1} AUC: {score:.4f}")
            mlflow.log_metric(f"fold_{fold+1}_auc", score)

        mean_auc = np.mean(auc_scores)
        logging.info(f"🏆 Mean CV AUC: {mean_auc:.4f}")
        mlflow.log_metric("mean_auc", mean_auc)

        # --- Phase 2: Production Retraining ---
        logging.info("🔄 Retraining Pipeline on FULL dataset...")
        pipeline.fit(X, y)
        
        # --- Phase 3: Save Artifacts ---
        model_dir = config['directories']['model_store']
        os.makedirs(model_dir, exist_ok=True)
        
        # Save .pkl (Standard Python format)
        local_model_path = os.path.join(model_dir, 'production_pipeline.pkl')
        joblib.dump(pipeline, local_model_path)
        logging.info(f"💾 Saved local pipeline to: {local_model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model", 
            input_example=X.head(1)
        )
        
        logging.info("✨ Training Complete.")

if __name__ == "__main__":
    train_model()
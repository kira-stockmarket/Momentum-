import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import optuna
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def train_and_evaluate():
    print("Phase 2: Loading dataset for ML Training...")
    
    if not os.path.exists('data/nifty100_features.parquet'):
        raise FileNotFoundError("Dataset not found. Run 1_download_data.py first.")
        
    df = pd.read_parquet('data/nifty100_features.parquet')
    
    features = [c for c in df.columns if c not in ['Target', 'Future_Return', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    X = df[features]
    y = df['Target']
    
    print(f"Starting Walk-Forward Optimization on {len(X)} records...")

    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 500),
            'verbose': -1
        }
        
        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            gbm = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            preds = gbm.predict(X_test)
            cv_scores.append(log_loss(y_test, preds))
            
        return np.mean(cv_scores)

    print("Running Hyperparameter Hunt for exactly 1 Hour (3600 seconds)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, timeout=3600, n_jobs=-1)
    
    final_params = study.best_params
    final_params.update({'objective': 'binary', 'verbose': -1})
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(final_params, train_data, num_boost_round=500)
    
    final_model.save_model('models/momentum_model.txt')
    
    print("Generating SHAP Analysis for exact pre-momentum setups...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X, show=False)
    plt.savefig('reports/shap_momentum_setup.png', bbox_inches='tight')
    plt.close()
    print("Pipeline Complete. Model and SHAP rules saved.")

if __name__ == "__main__":
    train_and_evaluate()

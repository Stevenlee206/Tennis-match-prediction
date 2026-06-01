import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler


def plot_optuna_results(study : optuna.study.Study, reports_dir : Path):
    """
    Extract data and plot
    """
    df = study.trials_dataframe() # Extract to df
    # Take success trial
    df = df[df['state'] == 'COMPLETE']

    if df.empty:
        print("[!] There is no valid trial data to plot..")
        return

    trials = df['number']
    values = df['value']
    # cummin() helps create a line that retains the best (lowest) value up to the current time.
    best_values = values.cummin()
    plt.figure(figsize=(10, 6))

    # Each point correspond to a Trial
    plt.scatter(trials, values, alpha=0.6, color='teal', label='Trial Value (Log Loss)')

    # Draw a line to show the convergence process
    plt.plot(trials, best_values, color='red', linewidth=2.5, label='Best Value (Convergence)')

    plt.title('Optuna Optimization History\n(XGBoost Hyperparameter Tuning)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Trial Number', fontsize=11)
    plt.ylabel('Log Loss', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = reports_dir / "xgboost_optuna_tuning_history.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] The Optuna History chart has been saved at: {save_path.name}")

def apply_transforms(X_train, X_val=None):
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)
    X_val_proc = scaler.transform(X_val) if X_val is not None else None
    return X_train_proc, X_val_proc, scaler

def run_xgboost_optuna_pipeline(X_train, y_train, X_val, y_val,
                                output_dir, reports_dir,
                                n_trials=30,validation="holdout", n_splits=5,
                                tscv_test_size=None, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_model.joblib"
    scaler_path = output_dir / "xgboost_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Find model at {output_dir.name}. Skip training!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Start optimize xgboost by Optuna ({n_trials} trials) ---")

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),
            'enable_categorical': True,
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'random_state': 42
        }

        if X_val is not None:
            # Holdout
            X_tr_proc, X_v_proc, _ = apply_transforms(X_train, X_val)

            model = XGBClassifier(**param)
            model.fit(X_tr_proc, y_train)
            preds = model.predict_proba(X_v_proc)
            return log_loss(y_val, preds)

        else:
            # Walk-Forward (Time-Series CV)
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                X_tr_proc, X_v_proc, _ = apply_transforms(X_tr, X_v)

                model = XGBClassifier(**param)
                model.fit(X_tr_proc, y_tr)

                preds = model.predict_proba(X_v_proc)
                loss = log_loss(y_v, preds)
                cv_scores.append(loss)

            return np.mean(cv_scores)

    # Run Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"-> Best parameters from Optuna: {best_params}")

    # Plot
    plot_optuna_results(study, reports_dir)

    # Train final model
    print("\n Training final model with best params...")
    if validation == "holdout" and X_val is not None:
        X_final_raw = pd.concat([X_train, X_val])
        y_final = pd.concat([y_train, y_val])
    else:
        X_final_raw, y_final = X_train, y_train

    X_final_proc, _, final_scaler = apply_transforms(X_final_raw, None)

    final_clf = XGBClassifier(**best_params,
                              tree_method='hist', eval_metric='logloss',
                              random_state=42)
    final_clf.fit(X_final_proc, y_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(final_scaler, scaler_path)

    config = {
        "model_type": "XGBoost_Optuna",
        "best_params": best_params
    }
    with open(output_dir / "xgboost_config.json", "w") as f:
        json.dump(config, f, indent=4)

    return final_clf, final_scaler
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import FunctionTransformer


def plot_optuna_results(study, reports_dir):
    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']

    if df.empty:
        return

    trials = df['number']
    values = df['value']
    best_values = values.cummin()

    plt.figure(figsize=(10, 6))
    plt.scatter(trials, values, alpha=0.6, color='teal', label='Trial Value (Log Loss)')
    plt.plot(trials, best_values, color='red', linewidth=2.5, label='Best Value (Hội tụ)')

    plt.title('Optuna Optimization History\n(Decision Tree Tuning)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Trial Number', fontsize=11)
    plt.ylabel('Log Loss', fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = reports_dir / "dt_optuna_tuning_history.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] The Optuna History chart has been saved at: {save_path.name}")


def run_decision_tree_optuna_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, **kwargs):
    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "decisiontree_model.joblib"
    scaler_path = output_dir / "decisiontree_scaler.joblib"

    force_retrain = kwargs.get('force_retrain', False)
    if not force_retrain and model_path.exists() and scaler_path.exists():
        print(f"\n[!] Find model at {output_dir.name}. Skip training !")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"\n--- Start optimizing your Decision Tree with Optuna. ({n_trials} trials) ---")

    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': 42
        }

        if X_val is not None:
            model = DecisionTreeClassifier(**param)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            return log_loss(y_val, preds)
        else:
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = DecisionTreeClassifier(**param)
                model.fit(X_tr, y_tr)
                preds = model.predict_proba(X_v)
                cv_scores.append(log_loss(y_v, preds))
            return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"-> Best parameters : {best_params}")
    plot_optuna_results(study, reports_dir)

    print("\n Train final model...")
    if X_val is not None:
        X_final, y_final = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    else:
        X_final, y_final = X_train, y_train

    final_clf = DecisionTreeClassifier(**best_params, random_state=42)
    final_clf.fit(X_final, y_final)

    dummy_scaler = FunctionTransformer(func=None)
    dummy_scaler.fit(X_final)

    joblib.dump(final_clf, model_path)
    joblib.dump(dummy_scaler, scaler_path)

    with open(output_dir / "decisiontree_config.json", "w") as f:
        json.dump({"model_type": "DecisionTree_Optuna", "best_params": best_params}, f, indent=4)

    return final_clf, dummy_scaler
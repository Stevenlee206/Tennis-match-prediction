import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, kernel):
    # Base params
    c_param = trial.suggest_float("C", 1e-3, 1e2, log=True)
    params = {"C": c_param, "kernel": kernel, "random_state": 42}
    
    # Kernel specific params
    if kernel in ['rbf', 'poly']:
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

    clf = SVC(**params)
    clf.fit(X_train_scaled, y_train)
    val_preds = clf.predict(X_val_scaled)
    return accuracy_score(y_val, val_preds)

def plot_optuna_history(study, save_path):
    plt.figure(figsize=(10, 6))
    trials = study.trials_dataframe()
    if not trials.empty and "value" in trials.columns:
        sns.lineplot(data=trials, x="number", y="value", marker="o")
        plt.title("Optuna Optimization History (Validation Accuracy)")
        plt.xlabel("Trial Number")
        plt.ylabel("Validation Accuracy")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path / "optuna_optimization_history.png", dpi=300)
    plt.close()

def plot_feature_importance(clf, feature_names, save_path):
    importances = clf.coef_[0]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': importances,
        'Absolute_Importance': np.abs(importances)
    }).sort_values(by='Absolute_Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Coefficient', y='Feature', palette="vlag")
    plt.title("SVM Feature Importance (Linear Coefficients)")
    plt.xlabel("Coefficient Value (Directional Impact)")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

def run_svm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, kernel="linear"):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "svm_model.joblib"
    scaler_path = output_dir / "svm_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training and inferring immediately!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "svm_sklearn_optuna.db"
    storage_url = f"sqlite:///{db_path.absolute()}"
    
    print(f"\nStarting Optuna hyperparameter search ({n_trials} trials for {kernel.upper()} kernel)...")
    study = optuna.create_study(
        study_name="svm_optimization",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, kernel),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    print("Training final model with optimal parameters...")
    final_clf = SVC(**best_params, kernel=kernel, random_state=42)
    final_clf.fit(X_train_scaled, y_train)
    
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "C-SVM",
        "kernel": kernel,
        "best_params": best_params,
        "val_accuracy": study.best_value,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "svm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_optuna_history(study, reports_dir)
    
    # Feature Importance (coef_) is strictly for linear kernels
    if kernel == "linear":
        plot_feature_importance(final_clf, X_train.columns, reports_dir)
    else:
        print(f"[!] Skipping Feature Importance Plot: 'coef_' is not available for the {kernel.upper()} kernel.")
    
    return final_clf, scaler
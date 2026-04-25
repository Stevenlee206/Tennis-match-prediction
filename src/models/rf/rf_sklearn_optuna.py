import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, n_est_min, n_est_max, depth_min, depth_max):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", n_est_min, n_est_max),
        "max_depth": trial.suggest_int("max_depth", depth_min, depth_max),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "random_state": 42,
        "n_jobs": -1
    }
    
    clf = RandomForestClassifier(**params)
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
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis")
    plt.title("Random Forest Feature Importance (Gini Impurity Decrease)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

def run_rf_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, n_est_min=50, n_est_max=500, depth_min=5, depth_max=50):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "rf_model.joblib"
    scaler_path = output_dir / "rf_scaler.joblib"

    print("Scaling features... (Optional for RF, kept for pipeline consistency)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "rf_sklearn_optuna.db"
    
    study_name = f"rf_optimization_{output_dir.name}"
    
    print(f"\nStarting Optuna hyperparameter search ({n_trials} trials)...")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path.absolute()}",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, n_est_min, n_est_max, depth_min, depth_max),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    final_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_scaled, y_train)
    
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "RandomForest",
        "optimizer": "Optuna",
        "best_params": best_params,
        "val_accuracy": study.best_value,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "rf_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_optuna_history(study, reports_dir)
    plot_feature_importance(final_clf, X_train.columns, reports_dir)
    
    return final_clf, scaler
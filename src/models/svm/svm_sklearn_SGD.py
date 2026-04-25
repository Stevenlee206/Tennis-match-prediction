import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# Optimization Objective
# ==========================================
def objective(trial, X_train_scaled, y_train, X_val_scaled, y_val):
    """
    Optuna objective function to tune SGD specifically.
    Instead of tuning C for LIBSVM, we tune alpha directly for SGD.
    """
    alpha_param = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    # We can also let Optuna tune the initial learning rate!
    eta0_param = trial.suggest_float("eta0", 1e-4, 1e-1, log=True)
    
    clf = SGDClassifier(
        loss="hinge", 
        penalty="l2", 
        alpha=alpha_param, 
        learning_rate="adaptive", 
        eta0=eta0_param,
        max_iter=200, # Sufficient iterations for validation approximation
        random_state=42
    )
    
    clf.fit(X_train_scaled, y_train)
    val_preds = clf.predict(X_val_scaled)
    
    return accuracy_score(y_val, val_preds)

# ==========================================
# Plotting Utilities
# ==========================================
def plot_optuna_history(study, save_path):
    plt.figure(figsize=(10, 6))
    trials = study.trials_dataframe()
    if not trials.empty and "value" in trials.columns:
        sns.lineplot(data=trials, x="number", y="value", marker="o")
        plt.title("Optuna Optimization History (Validation Accuracy)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path / "optuna_optimization_history_svm.png", dpi=300)
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
    plt.title("SGD-SVM Feature Importance (Linear Coefficients)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

# ==========================================
# Main Execution Pipeline
# ==========================================
def run_svm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, n_epochs=100, kernel="linear"):
    
    if kernel != "linear":
        raise ValueError("SGDClassifier requires a linear kernel. Terminating.")    
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    ckpt_dir = output_dir / "checkpoints"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "svm_model.joblib"
    scaler_path = output_dir / "svm_scaler.joblib"

    # --- INFERENCE SHORTCUT ---
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
    
    print(f"\nStarting Optuna search ({n_trials} trials)...")
    study = optuna.create_study(
        study_name="svm_optimization_sgd",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )
    
    completed = len(study.trials)
    remaining = max(0, n_trials - completed)
    if remaining > 0:
        study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val), n_trials=remaining)
        
    best_params = study.best_params
    print(f"\nBest Optuna parameters found: {best_params}")
    
    existing_ckpts = list(ckpt_dir.glob("svm_epoch_*.joblib"))
    start_epoch = 0
    
    if existing_ckpts:
        latest_ckpt = max(existing_ckpts, key=os.path.getctime)
        start_epoch = int(latest_ckpt.stem.split("_")[-1])
        print(f"Resuming final training from checkpoint: Epoch {start_epoch}")
        final_clf = joblib.load(latest_ckpt)
    else:
        print("Starting final training from scratch with mini-batches...")
        final_clf = SGDClassifier(
            loss="hinge", 
            penalty="l2", 
            alpha=best_params["alpha"], 
            learning_rate="adaptive",
            eta0=best_params["eta0"],
            random_state=42
        )

    classes = np.unique(y_train)
    n_samples = X_train_scaled.shape[0]
    batch_size = 64 # Mini-batch size

    # Safety cast for pandas Series
    y_train_arr = y_train.values if hasattr(y_train, "values") else y_train

    # Epoch Training Loop (True Stochastic/Mini-Batch Gradient Descent)
    for epoch in range(start_epoch + 1, n_epochs + 1):
        # 1. Shuffle data at the beginning of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train_scaled[indices]
        y_shuffled = y_train_arr[indices]
        
        # 2. Train in mini-batches
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            final_clf.partial_fit(X_batch, y_batch, classes=classes)
            
        # Save checkpoints every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs:
            joblib.dump(final_clf, ckpt_dir / f"svm_epoch_{epoch}.joblib")
            print(f"Epoch {epoch}/{n_epochs} completed and saved.")

    # Save final models
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "Linear-SVM (SGD)",
        "best_alpha": best_params["alpha"],
        "best_eta0": best_params["eta0"],
        "epochs_trained": n_epochs,
        "batch_size": batch_size,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "svm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    plot_optuna_history(study, reports_dir)
    plot_feature_importance(final_clf, X_train.columns, reports_dir)
    
    return final_clf, scaler
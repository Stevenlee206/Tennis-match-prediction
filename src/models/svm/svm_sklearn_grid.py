import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
from sklearn.exceptions import ConvergenceWarning

# ==========================================
# Plotting Utilities
# ==========================================
def plot_grid_history(c_values, accuracies, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=c_values, y=accuracies, marker="o", color='teal')
    plt.xscale('log') # Log scale is vital for visualizing the C parameter
    plt.title("Nested Grid Search Optimization (Inner CV Accuracy vs. C)")
    plt.xlabel("C Value (Log Scale)")
    plt.ylabel("Inner CV Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "grid_search_history.png", dpi=300)
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
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

# ==========================================
# Main Execution Pipeline
# ==========================================
def run_svm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, c_min=1e-3, c_max=1e2, c_steps=10, kernel="linear"):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "svm_model.joblib"
    scaler_path = output_dir / "svm_scaler.joblib"

    # 1. Combine Train and Val for Inner Cross-Validation
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # 2. Define the discrete grid
    c_grid = np.logspace(np.log10(c_min), np.log10(c_max), num=c_steps)
    param_grid = {'C': c_grid}

    print(f"\n--- Starting Nested GridSearchCV ({c_steps} Steps from {c_min} to {c_max}) ---")
    
    # 3. Use TimeSeriesSplit for the Inner Loop
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=SVC(kernel=kernel, max_iter=20000, random_state=42),
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_full_scaled, y_train_full)

    best_params = grid_search.best_params_
    best_acc = grid_search.best_score_
    final_clf = grid_search.best_estimator_

    print(f"\nOptimization Complete.")
    print(f"Best parameter found (C): {best_params['C']:.6f} with Inner CV Accuracy: {best_acc:.4f}")
    
    # 4. Save Artifacts
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "C-SVM",
        "optimizer": "Nested_GridSearch_TimeSeriesSplit",
        "kernel": kernel,
        "c_min": c_min,
        "c_max": c_max,
        "c_steps": c_steps,
        "best_C": best_params['C'],
        "inner_cv_accuracy": best_acc,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "svm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    # Extract GridSearchCV results to plot the history curve
    cv_results = grid_search.cv_results_
    plot_grid_history(c_grid, cv_results['mean_test_score'], reports_dir)
    
    if kernel == "linear":
        plot_feature_importance(final_clf, X_train.columns, reports_dir)
    else:
        print(f"[!] Skipping Feature Importance Plot: 'coef_' is not available for the {kernel.upper()} kernel.")
    
    return final_clf, scaler
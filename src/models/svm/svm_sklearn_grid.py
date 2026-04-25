import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

# ==========================================
# Plotting Utilities
# ==========================================
def plot_grid_history(c_values, accuracies, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=c_values, y=accuracies, marker="o", color='teal')
    plt.xscale('log') # Log scale is vital for visualizing the C parameter
    plt.title("Grid Search Optimization (Validation Accuracy vs. C)")
    plt.xlabel("C Value (Log Scale)")
    plt.ylabel("Validation Accuracy")
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

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Silence Convergence warnings during tuning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Generate the grid in logarithmic space
    c_grid = np.logspace(np.log10(c_min), np.log10(c_max), num=c_steps)
    
    best_c = None
    best_val_acc = -1
    history_accuracies = []

    print(f"\n--- Starting Grid Search ({c_steps} Steps from {c_min} to {c_max}) ---")
    
    for i, c_val in enumerate(c_grid):
        print(f"  -> Evaluating Step {i+1:02d}/{c_steps} [C = {c_val:10.5f}]... ", end="", flush=True)
        
        clf = SVC(C=c_val, kernel=kernel, max_iter=20000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val_scaled))
        
        history_accuracies.append(acc)
        print(f"Acc: {acc:.4f}")
        
        if acc > best_val_acc:
            best_val_acc = acc
            best_c = c_val

    print(f"\nOptimization Complete.")
    print(f"Best parameter found (C): {best_c:.6f} with Accuracy: {best_val_acc:.4f}")
    
    print(f"Training final model with optimal parameters ({kernel} kernel)...")
    final_clf = SVC(C=best_c, kernel=kernel, random_state=42)
    final_clf.fit(X_train_scaled, y_train)
    
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "C-SVM",
        "optimizer": "GridSearch",
        "kernel": kernel,
        "c_min": c_min,
        "c_max": c_max,
        "c_steps": c_steps,
        "best_C": best_c,
        "val_accuracy": best_val_acc,
        "features_used": list(X_train.columns)
    }
    with open(output_dir / "svm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_grid_history(c_grid, history_accuracies, reports_dir)
    
    if kernel == "linear":
        plot_feature_importance(final_clf, X_train.columns, reports_dir)
    else:
        print(f"[!] Skipping Feature Importance Plot: 'coef_' is not available for the {kernel.upper()} kernel.")
    
    return final_clf, scaler
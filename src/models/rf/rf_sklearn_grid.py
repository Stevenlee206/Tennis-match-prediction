import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def plot_grid_heatmap(results_df, save_path):
    pivot_table = results_df.pivot(index="max_depth", columns="n_estimators", values="accuracy")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Validation Accuracy'})
    plt.title("Grid Search Heatmap: Accuracy vs. Hyperparameters")
    plt.gca().invert_yaxis() # Standardize origin layout
    plt.tight_layout()
    plt.savefig(save_path / "grid_search_heatmap.png", dpi=300)
    plt.close()

def plot_feature_importance(clf, feature_names, save_path):
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

def run_rf_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_est_min=50, n_est_max=500, n_est_steps=5, depth_min=5, depth_max=50, depth_steps=5):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "rf_model.joblib"
    scaler_path = output_dir / "rf_scaler.joblib"

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Dynamically define the discrete grid based on user input
    n_estimators_grid = sorted(list(set(np.linspace(n_est_min, n_est_max, n_est_steps, dtype=int).tolist())))
    max_depth_grid = sorted(list(set(np.linspace(depth_min, depth_max, depth_steps, dtype=int).tolist())))
    
    best_acc = -1
    best_params = {}
    results = []

    print(f"\n--- Starting 2D Grid Search ({len(n_estimators_grid) * len(max_depth_grid)} combinations) ---")
    
    for n_est in n_estimators_grid:
        for depth in max_depth_grid:
            print(f"  -> Evaluating [n_estimators={n_est}, max_depth={depth}]... ", end="", flush=True)
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
            clf.fit(X_train_scaled, y_train)
            acc = accuracy_score(y_val, clf.predict(X_val_scaled))
            print(f"Acc: {acc:.4f}")
            
            results.append({"n_estimators": n_est, "max_depth": depth, "accuracy": acc})
            
            if acc > best_acc:
                best_acc = acc
                best_params = {"n_estimators": n_est, "max_depth": depth}

    print(f"\nBest parameters: {best_params} with Accuracy: {best_acc:.4f}")
    
    final_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_clf.fit(X_train_scaled, y_train)
    
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "RandomForest",
        "optimizer": "GridSearch",
        "grid_params": {
            "n_estimators_grid": n_estimators_grid,
            "max_depth_grid": max_depth_grid,
        },
        "best_params": best_params,
        "val_accuracy": best_acc,
        "features_used": list(X_train.columns),
    }
    with open(output_dir / "rf_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    plot_grid_heatmap(pd.DataFrame(results), reports_dir)
    plot_feature_importance(final_clf, X_train.columns, reports_dir)
    
    return final_clf, scaler
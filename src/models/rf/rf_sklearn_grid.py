import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.utils.paths import ensure_writable_path

def plot_grid_heatmap(results_df, save_path):
    pivot_table = results_df.pivot(index="max_depth", columns="n_estimators", values="accuracy")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Validation Accuracy'})
    plt.title("Nested Grid Search Heatmap: Accuracy vs. Hyperparameters")
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
    output_dir = ensure_writable_path(output_dir)
    reports_dir = ensure_writable_path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "rf_model.joblib"
    scaler_path = output_dir / "rf_scaler.joblib"

    # 1. Combine Train and Val for the Inner Cross-Validation
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)

    # 2. Define the discrete grid
    n_estimators_grid = sorted(list(set(np.linspace(n_est_min, n_est_max, n_est_steps, dtype=int).tolist())))
    max_depth_grid = sorted(list(set(np.linspace(depth_min, depth_max, depth_steps, dtype=int).tolist())))
    
    param_grid = {
        "n_estimators": n_estimators_grid,
        "max_depth": max_depth_grid
    }

    print(f"\n--- Starting Nested GridSearchCV ({len(n_estimators_grid) * len(max_depth_grid)} combinations) ---")
    
    # 3. Use TimeSeriesSplit for the Inner Loop to prevent future data leakage
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
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

    print(f"\nBest parameters: {best_params} with Inner CV Accuracy: {best_acc:.4f}")
    
    # 4. Save Artifacts
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "RandomForest",
        "optimizer": "Nested_GridSearch_TimeSeriesSplit",
        "grid_params": {
            "n_estimators_grid": n_estimators_grid,
            "max_depth_grid": max_depth_grid,
        },
        "best_params": best_params,
        "inner_cv_accuracy": best_acc,
        "features_used": list(X_train.columns),
    }
    with open(output_dir / "rf_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # 5. Extract GridSearchCV results to rebuild the Heatmap
    cv_results = pd.DataFrame(grid_search.cv_results_)
    results_df = pd.DataFrame({
        "n_estimators": cv_results["param_n_estimators"].astype(int),
        "max_depth": cv_results["param_max_depth"].astype(int),
        "accuracy": cv_results["mean_test_score"]
    })

    plot_grid_heatmap(results_df, reports_dir)
    plot_feature_importance(final_clf, X_train.columns, reports_dir)
    
    return final_clf, scaler
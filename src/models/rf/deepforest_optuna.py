import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path

from deepforest import CascadeForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# ==========================================
# Plotting Utilities
# ==========================================
np.int = int      
np.float = float  
np.bool = bool    
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
        plt.savefig(save_path / "optuna_optimization_history_deepforest.png", dpi=300)
    plt.close()

def plot_feature_importance(model, X, y, feature_names, save_path):
    print("Calculating permutation importance (this may take a moment)...")
    
    # Use permutation importance since Deep Forest doesn't expose native importances
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis")
    plt.title("Deep Forest Feature Importance (Permutation)")
    plt.xlabel("Mean Accuracy Decrease (Higher = More Important)")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "deepforest_feature_importance.png", dpi=300)
    plt.close()

# ==========================================
# Weight Generation (Matched Logic)
# ==========================================
def generate_sample_weights(X_raw, y_raw, strategy="none", base_weight=1.0):
    n_samples = len(y_raw)
    weights = np.ones(n_samples)
    
    if strategy == "none" or base_weight <= 1.0 or 'elo_diff' not in X_raw.columns:
        return weights

    y_vals = y_raw.values if isinstance(y_raw, pd.Series) else y_raw
    elo_diffs = X_raw['elo_diff'].values
    upset_mask = ((elo_diffs > 0) & (y_vals == 0)) | ((elo_diffs < 0) & (y_vals == 1))

    if strategy == "static":
        weights[upset_mask] = base_weight
    elif strategy == "magnitude":
        for i in range(n_samples):
            if upset_mask[i]:
                gap_severity = abs(elo_diffs[i]) / 100.0
                weights[i] = 1.0 + (base_weight * gap_severity)
    elif strategy == "temporal":
        decay_curve = np.exp(np.linspace(-3, 0, n_samples))
        for i in range(n_samples):
            if upset_mask[i]:
                weights[i] = 1.0 + ((base_weight - 1.0) * decay_curve[i])

    return weights

# ==========================================
# Optimization Objective
# ==========================================
def objective(trial, X_train, y_train, X_val, y_val, bounds, add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0, n_splits=3, tscv_test_size=None):    
    # Deep Forest Hyperparameters using dynamic bounds from CLI
    params = {
        "n_bins": trial.suggest_int("n_bins", 128, 255),
        "max_layers": trial.suggest_int("max_layers", bounds['max_layers_min'], bounds['max_layers_max']),
        "n_estimators": trial.suggest_int("n_estimators", 2, 4), 
        "n_trees": trial.suggest_int("n_trees", bounds['n_trees_min'], bounds['n_trees_max']), 
        "max_depth": trial.suggest_int("max_depth", bounds['max_depth_min'], bounds['max_depth_max']),
        "min_samples_split": trial.suggest_int("min_samples_split", 15, 50),
        "random_state": 42,
        "n_jobs": -1
    }

    # ==========================================
    # STRATEGY 1: STANDARD HOLDOUT
    # ==========================================
    if validation == "holdout":
        scaler = StandardScaler()
        X_t_scaled = scaler.fit_transform(X_train)
        X_v_scaled = scaler.transform(X_val)
        
        if add_pca:
            pca = PCA(n_components=0.95, random_state=42)
            X_t_processed = pca.fit_transform(X_t_scaled)
            X_v_processed = pca.transform(X_v_scaled)
        else:
            X_t_processed, X_v_processed = X_t_scaled, X_v_scaled
            
        weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
        
        clf = CascadeForestClassifier(**params)
        clf.fit(X_t_processed, y_train.values, sample_weight=weights)
        
        val_preds = clf.predict(X_v_processed)
        return accuracy_score(y_val, val_preds)

    # ==========================================
    # STRATEGY 2: WALK-FORWARD (TSCV)
    # ==========================================
    elif validation == "walk_forward":
        # ---> UPDATED HERE <---
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)
        fold_accuracies = []
        
        for train_index, val_index in tscv.split(X_train):
            X_t_cv, X_v_cv = X_train.iloc[train_index], X_train.iloc[val_index]
            y_t_cv, y_v_cv = y_train.iloc[train_index], y_train.iloc[val_index]
            
            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t_cv)
            X_v_scaled = scaler.transform(X_v_cv)
            
            if add_pca:
                pca = PCA(n_components=0.95, random_state=42)
                X_t_processed = pca.fit_transform(X_t_scaled)
                X_v_processed = pca.transform(X_v_scaled)
            else:
                X_t_processed, X_v_processed = X_t_scaled, X_v_scaled
                
            weights_cv = generate_sample_weights(X_t_cv, y_t_cv, weight_strategy, upset_weight)
            
            clf_cv = CascadeForestClassifier(**params)
            clf_cv.fit(X_t_processed, y_t_cv.values, sample_weight=weights_cv)
            
            val_preds = clf_cv.predict(X_v_processed)
            fold_accuracies.append(accuracy_score(y_v_cv, val_preds))
            
        return np.mean(fold_accuracies)

# ==========================================
# Main Execution Pipeline
# ==========================================
def run_deepforest_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0, 
                            max_layers_min=2, max_layers_max=8, n_trees_min=50, n_trees_max=200, max_depth_min=5, max_depth_max=30, n_splits=3, tscv_test_size=None):
    
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "deepforest_model.joblib"
    scaler_path = output_dir / "deepforest_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print("Scaling features for Deep Forest...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if add_pca:
        print("Applying PCA (Retaining 95% variance)...")
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        joblib.dump(pca, output_dir / "deepforest_pca.joblib")
    else:
        X_train_processed = X_train_scaled

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "deepforest_optuna.db"
    
    # Pack bounds into a dictionary for clean passing to the objective
    bounds = {
        'max_layers_min': max_layers_min, 'max_layers_max': max_layers_max,
        'n_trees_min': n_trees_min, 'n_trees_max': n_trees_max,
        'max_depth_min': max_depth_min, 'max_depth_max': max_depth_max
    }
    
    print(f"\nStarting Optuna search for Deep Forest ({n_trials} trials | Upset Strategy: {weight_strategy.upper()})...")
    study = optuna.create_study(
        study_name="deepforest_optimization",
        storage=f"sqlite:///{db_path.absolute()}",
        load_if_exists=True,
        direction="maximize"
    )
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, bounds, add_pca, validation, weight_strategy, upset_weight, n_splits, tscv_test_size), n_trials=n_trials)
        
    best_params = study.best_params
    print(f"\nBest Optuna parameters: {best_params}")
    
    # --- Final Training ---
    print("\nTraining final Deep Forest architecture...")
    final_clf = CascadeForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    final_weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
    final_clf.fit(X_train_processed, y_train.values, sample_weight=final_weights)
    
    # OVERFITTING CHECK
    train_preds = final_clf.predict(X_train_processed)
    train_acc = accuracy_score(y_train, train_preds)
    
    print("\n" + "-"*30)
    print(" OVERFITTING CHECK")
    print("-"*30)
    print(f"Training Accuracy:     {train_acc * 100:.2f}%")
    print(f"Optuna Val Accuracy:   {study.best_value * 100:.2f}%")
    
    # Deep Forest tracks its own internal layer-by-layer validation
    print(f"Total Cascade Layers:  {final_clf.n_layers_}")
    print("-" * 30 + "\n")

    # Generate Plots
    print("Generating optimization and importance plots...")
    plot_optuna_history(study, reports_dir)
    
    if add_pca:
        final_feature_names = [f"PC{i+1}" for i in range(X_train_processed.shape[1])]
    else:
        final_feature_names = list(X_train.columns)
        
    plot_feature_importance(final_clf, X_train_processed, y_train.values, final_feature_names, reports_dir)
    # Save
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "gcForest",
        "best_params": best_params,
        "val_accuracy": study.best_value,
        "train_accuracy": train_acc,
        "pca_applied": add_pca,
        "cascade_layers_built": final_clf.n_layers_,
        "weight_strategy": weight_strategy,
        "upset_weight": upset_weight,
        "features_used": final_feature_names
    }
    with open(output_dir / "deepforest_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    return final_clf, scaler
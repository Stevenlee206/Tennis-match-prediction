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
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def generate_sample_weights(X_raw, y_raw, strategy="none", base_weight=1.0):
    """
    Dynamically calculates sample weights based on upset severity.
    """
    n_samples = len(y_raw)
    weights = np.ones(n_samples)
    
    if strategy == "none" or base_weight <= 1.0 or 'elo_diff' not in X_raw.columns:
        return weights

    y_vals = y_raw.values if isinstance(y_raw, pd.Series) else y_raw
    elo_diffs = X_raw['elo_diff'].values
    
    # Base Upset Mask
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

def objective(trial, X_train, y_train, X_val, y_val, kernel, c_min=1e-3, c_max=1e2, add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0, n_splits=5, tscv_test_size=None):
    c_param = trial.suggest_float("C", c_min, c_max, log=True)
    params = {"C": c_param, "kernel": kernel, "random_state": 42}
    
    # Kernel specific params
    if kernel in ['rbf', 'poly']:
        params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

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
        
        # Enable probability=True for scoring metrics
        clf = SVC(**params, probability=True)
        clf.fit(X_t_processed, y_train, sample_weight=weights)
        val_preds = clf.predict(X_v_processed)
        
        return accuracy_score(y_val, val_preds)
        
    # ==========================================
    # STRATEGY 2: WALK-FORWARD (TSCV)
    # ==========================================
    elif validation == "walk_forward":
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)
        fold_accuracies = []
        
        for train_index, val_index in tscv.split(X_train):
            # Use .copy() to prevent SettingWithCopyWarnings
            X_t_cv = X_train.iloc[train_index].copy()
            X_v_cv = X_train.iloc[val_index].copy()
            y_t_cv = y_train.iloc[train_index].copy()
            y_v_cv = y_train.iloc[val_index].copy()
            
            # ---> FILTERING LOGIC FOR AUGMENTED DATA <---
            if 'is_augmented' in X_v_cv.columns:
                val_mask = (X_v_cv['is_augmented'] == 0)
                X_v_cv = X_v_cv[val_mask]
                y_v_cv = y_v_cv[val_mask]
                
                X_t_cv = X_t_cv.drop(columns=['is_augmented'])
                X_v_cv = X_v_cv.drop(columns=['is_augmented'])
            
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
            
            # Enable probability=True for scoring metrics
            clf_cv = SVC(**params, probability=True)
            clf_cv.fit(X_t_processed, y_t_cv, sample_weight=weights_cv)
            val_preds = clf_cv.predict(X_v_processed)
            
            fold_accuracies.append(accuracy_score(y_v_cv, val_preds))
            
        return np.mean(fold_accuracies)


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


def run_svm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, kernel="linear", c_min=1e-3, c_max=1e2, add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0, n_splits=5, tscv_test_size=None):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{kernel}_model.joblib"
    scaler_path = output_dir / f"{kernel}_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training and inferring immediately!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print(f"Scaling features for SVM ({kernel.upper()})...")
    
    # Protect scaler from the augmented metadata flag
    if 'is_augmented' in X_train.columns:
        X_train_features = X_train.drop(columns=['is_augmented'])
    else:
        X_train_features = X_train
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    if add_pca:
        print(f"Applying PCA for SVM (Retaining 95% variance)...")
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        pca_path = output_dir / f"{kernel}_pca.joblib"
        joblib.dump(pca, pca_path)
        
        if X_val_scaled is not None:
            X_val_processed = pca.transform(X_val_scaled)
    else:
        X_train_processed = X_train_scaled
        if X_val_scaled is not None:
            X_val_processed = X_val_scaled

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / f"svm_{kernel}_optuna.db"
    storage_url = f"sqlite:///{db_path.absolute()}"
    
    print(f"\nStarting Optuna search ({n_trials} trials | {kernel.upper()} | C bounds: [{c_min}, {c_max}])...")
    study = optuna.create_study(
        study_name=f"svm_{kernel}_optimization",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, kernel, c_min, c_max, add_pca, validation, weight_strategy, upset_weight, n_splits, tscv_test_size),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    print("Training final model with optimal parameters...")
    # Enable probability=True for scoring metrics
    final_clf = SVC(**best_params, kernel=kernel, random_state=42, probability=True)
    
    final_weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
    final_clf.fit(X_train_processed, y_train, sample_weight=final_weights)
    
    train_preds = final_clf.predict(X_train_processed)
    train_acc = accuracy_score(y_train, train_preds)
    
    print("\n" + "-"*30)
    print(" OVERFITTING CHECK")
    print("-"*30)
    print(f"Training Accuracy:     {train_acc * 100:.2f}%")
    print(f"Optuna Val Accuracy:   {study.best_value * 100:.2f}%")
    
    if (train_acc - study.best_value) > 0.10:
        print("⚠️ WARNING: High likelihood of overfitting. The model is memorizing the training data.")
    print("-" * 30 + "\n") 
    
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    if add_pca:
        n_pcs = X_train_processed.shape[1]
        final_feature_names = [f"PC{i+1}" for i in range(n_pcs)]
    else:
        # Pull strictly from the filtered features dataframe
        final_feature_names = list(X_train_features.columns)

    config = {
        "model_type": "C-SVM",
        "kernel": kernel,
        "best_params": best_params,
        "train_accuracy": train_acc,
        "val_accuracy": study.best_value,
        "pca_applied": add_pca,
        "weight_strategy": weight_strategy,
        "upset_weight": upset_weight,
        "features_used": final_feature_names
    }
    with open(output_dir / f"{kernel}_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_optuna_history(study, reports_dir)
    
    if kernel == "linear":
        plot_feature_importance(final_clf, final_feature_names, reports_dir)
    else:
        print(f"[!] Skipping Feature Importance Plot: 'coef_' is not available for the {kernel.upper()} kernel.")
    
    return final_clf, scaler
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Weight Generation
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

# Optimization Objective
def objective(trial, X_train, y_train, X_val, y_val, c_min=1e-3, c_max=1e2, add_pca=False, add_kmeans=False, n_clusters=5, validation="holdout", weight_strategy="none", upset_weight=1.0, lr_schedule="adaptive", n_splits=5, tscv_test_size=None):
    c_param = trial.suggest_float("C", c_min, c_max, log=True)
    alpha_param = 1.0 / c_param 
    
    # Conditional Hyperparameter Search based on schedule
    if lr_schedule in ["constant", "invscaling", "adaptive"]:
        eta0_param = trial.suggest_float("eta0", 1e-4, 1e-1, log=True)
    else:
        eta0_param = 0.01 
        
    if lr_schedule == "invscaling":
        power_t_param = trial.suggest_float("power_t", 0.1, 1.0)
    else:
        power_t_param = 0.5 
        
    params = {
        "loss": "hinge", 
        "penalty": "l2", 
        "alpha": alpha_param, 
        "learning_rate": lr_schedule, 
        "eta0": eta0_param,
        "power_t": power_t_param,
        "max_iter": 200, 
        "random_state": 42
    }

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
        if add_kmeans:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            t_distances = kmeans.fit_transform(X_t_processed)
            v_distances = kmeans.transform(X_v_processed)
            X_t_processed = np.hstack((X_t_processed, t_distances))
            X_v_processed = np.hstack((X_v_processed, v_distances))  
        weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
        
        clf = SGDClassifier(**params)
        clf.fit(X_t_processed, y_train, sample_weight=weights)
        val_preds = clf.predict(X_v_processed)
        
        return accuracy_score(y_val, val_preds)

    elif validation == "walk_forward":
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
            if add_kmeans:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                t_distances = kmeans.fit_transform(X_t_processed)
                v_distances = kmeans.transform(X_v_processed)
                X_t_processed = np.hstack((X_t_processed, t_distances))
                X_v_processed = np.hstack((X_v_processed, v_distances))    
            weights_cv = generate_sample_weights(X_t_cv, y_t_cv, weight_strategy, upset_weight)
            
            clf_cv = SGDClassifier(**params)
            clf_cv.fit(X_t_processed, y_t_cv, sample_weight=weights_cv)
            val_preds = clf_cv.predict(X_v_processed)
            
            fold_accuracies.append(accuracy_score(y_v_cv, val_preds))
            
        return np.mean(fold_accuracies)

# Plotting Utilities
def plot_optuna_history(study, save_path):
    plt.figure(figsize=(10, 6))
    trials = study.trials_dataframe()
    if not trials.empty and "value" in trials.columns:
        sns.lineplot(data=trials, x="number", y="value", marker="o")
        plt.title("Optuna Optimization History (Validation Accuracy)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path / "optuna_optimization_history_sgd.png", dpi=300)
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


# Main Execution Pipeline
def run_svm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, n_epochs=100, kernel="linear", c_min=1e-3, c_max=1e2, add_pca=False, add_kmeans=False, n_clusters=5, validation="holdout", weight_strategy="none", upset_weight=1.0, lr_schedule="adaptive", n_splits=5, tscv_test_size=None):
    if kernel != "linear":
        raise ValueError("SGDClassifier requires a linear kernel. Terminating.")    
    
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "svm_sgd_model.joblib"
    scaler_path = output_dir / "svm_sgd_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training and inferring immediately!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    if add_pca:
        print(f"Applying PCA for SGD-SVM (Retaining 95% variance)...")
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        pca_path = output_dir / "svm_sgd_pca.joblib"
        joblib.dump(pca, pca_path)
        
        if X_val_scaled is not None:
            X_val_processed = pca.transform(X_val_scaled)
        else:
            X_val_processed = None
    else:
        X_train_processed = X_train_scaled
        X_val_processed = X_val_scaled
    if add_kmeans:
        print(f"Applying KMeans clustering (k={n_clusters}) for SGD-SVM...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        t_distances = kmeans.fit_transform(X_train_processed)
        X_train_processed = np.hstack((X_train_processed, t_distances))
        
        if X_val_processed is not None:
            v_distances = kmeans.transform(X_val_processed)
            X_val_processed = np.hstack((X_val_processed, v_distances))
            
        kmeans_path = output_dir / "svm_sgd_kmeans.joblib"
        joblib.dump(kmeans, kmeans_path)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "svm_sklearn_sgd_optuna.db"
    storage_url = f"sqlite:///{db_path.absolute()}"
    
    print(f"\nStarting Optuna search ({n_trials} trials | Schedule: {lr_schedule.upper()} | Upset Weight: {upset_weight}x)...")
    study = optuna.create_study(
        study_name="svm_optimization_sgd",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, c_min, c_max, add_pca, add_kmeans, n_clusters, validation, weight_strategy, upset_weight, lr_schedule, n_splits, tscv_test_size), n_trials=n_trials)
        
    best_params = study.best_params
    best_alpha = 1.0 / best_params["C"]
    
    best_eta0 = best_params.get("eta0", 0.01) 
    best_power_t = best_params.get("power_t", 0.5)

    print(f"\nBest Optuna parameters found: C={best_params['C']:.4f} (alpha={best_alpha:.6f})")
    
    # --- RESTORED NATIVE TRAINING ---
    print("Training final model natively to preserve Optuna accuracy...")
    final_clf = SGDClassifier(
        loss="hinge", 
        penalty="l2", 
        alpha=best_alpha, 
        learning_rate=lr_schedule,
        eta0=best_eta0,
        power_t=best_power_t,
        max_iter=200, # Matches Optuna
        random_state=42
    )

    y_train_arr = y_train.values if hasattr(y_train, "values") else y_train
    weights_full = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)

    # Train in one shot natively
    final_clf.fit(X_train_processed, y_train_arr, sample_weight=weights_full)

    # OVERFITTING DIAGNOSTIC 
    train_preds = final_clf.predict(X_train_processed)
    train_acc = accuracy_score(y_train, train_preds)

    print("\n" + "-"*30)
    print(" OVERFITTING CHECK")
    print("-"*30)
    print(f"Training Accuracy:     {train_acc * 100:.2f}%")
    print(f"Optuna Val Accuracy:   {study.best_value * 100:.2f}%")
    
    if (train_acc - study.best_value) > 0.10:
        print("Warning: High likelihood of overfitting. The model is memorizing the training data.")
    print("-" * 30 + "\n")

    # Plotting
    print("Generating plots...")
    plot_optuna_history(study, reports_dir)
    
    if add_pca:
        # Subtract the KMeans columns to get the true number of Principal Components
        n_pcs = X_train_processed.shape[1] - (n_clusters if add_kmeans else 0)
        final_feature_names = [f"PC{i+1}" for i in range(n_pcs)]
    else:
        final_feature_names = list(X_train.columns)
        
    # Append the new Clustering features to the name list so they match the coefficients
    if add_kmeans:
        kmeans_names = [f"KMeans_Dist_C{i+1}" for i in range(n_clusters)]
        final_feature_names.extend(kmeans_names)
        
    plot_feature_importance(final_clf, final_feature_names, reports_dir)
        
    plot_feature_importance(final_clf, final_feature_names, reports_dir)

    # Save final models
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "Linear-SVM (SGD)",
        "best_C": best_params["C"],
        "best_alpha": best_alpha,
        "lr_schedule": lr_schedule,
        "best_eta0": best_eta0,
        "val_accuracy": study.best_value,
        "pca_applied": add_pca,
        "kmeans_applied": add_kmeans,  # <--- NEW
        "n_clusters": n_clusters if add_kmeans else 0, # <--- NEW
        "weight_strategy": weight_strategy,
        "upset_weight": upset_weight,
        "features_used": final_feature_names
    }
    with open(output_dir / "svm_sgd_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    return final_clf, scaler

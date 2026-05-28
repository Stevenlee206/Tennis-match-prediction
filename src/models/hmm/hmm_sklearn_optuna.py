import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path
import warnings

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

# ==========================================
# STREAMLINED HMM CLASSIFIER
# ==========================================
class FastHMMClassifier:
    """
    A simplified, fast wrapper to use hmmlearn for classification.
    Removes the custom early-stopping loop to leverage C-level optimizations.
    """
    def __init__(self, n_components=3, covariance_type='diag', n_iter=100, min_covar=1e-3, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.random_state = random_state
        self.models = {}
        self.classes_ = []

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            
            model = GaussianHMM(
                n_components=min(self.n_components, max(1, len(X_c))), 
                covariance_type=self.covariance_type, 
                n_iter=self.n_iter, 
                min_covar=self.min_covar,
                random_state=self.random_state
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                if len(X_c) > 0:
                    model.fit(X_c)
                    
            self.models[c] = model
        return self

    def predict_proba(self, X):
        """
        Calculates raw probabilities by extracting log-likelihoods and 
        applying a stable softmax transformation.
        """
        log_likelihoods = np.full((X.shape[0], len(self.classes_)), -np.inf)
        
        # Score the independent observations efficiently
        for class_idx, class_label in enumerate(self.classes_):
            model = self.models.get(class_label)
            if model is not None:
                for i in range(X.shape[0]):
                    try:
                        # model.score returns the log-likelihood
                        log_likelihoods[i, class_idx] = model.score(X[i:i+1])
                    except Exception:
                        pass
        
        # Apply LogSumExp / Softmax trick for numerical stability
        max_logits = np.max(log_likelihoods, axis=1, keepdims=True)
        # Prevent completely -inf rows from turning into NaNs
        max_logits[max_logits == -np.inf] = 0.0 
        
        stabilized_exp = np.exp(log_likelihoods - max_logits)
        probabilities = stabilized_exp / np.sum(stabilized_exp, axis=1, keepdims=True)
        
        return probabilities

    def predict(self, X):
        """
        Returns the class with the highest probability.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def objective(trial, X_train, y_train, X_val, y_val, add_pca=False, validation="holdout", n_splits=5, tscv_test_size=None):
    # HMM Hyperparameters
    n_components = trial.suggest_int("n_components", 2, 6)
    covariance_type = trial.suggest_categorical("covariance_type", ["diag", "spherical"])
    n_iter = trial.suggest_int("n_iter", 50, 200)
    min_covar = trial.suggest_float("min_covar", 1e-4, 1e-2, log=True)
    
    params = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "n_iter": n_iter,
        "min_covar": min_covar,
        "random_state": 42
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
            
        clf = FastHMMClassifier(**params)
        clf.fit(X_t_processed, y_train)
        val_preds = clf.predict(X_v_processed)
        
        return accuracy_score(y_val, val_preds)
        
    # ==========================================
    # STRATEGY 2: WALK-FORWARD (TSCV)
    # ==========================================
    elif validation == "walk_forward":
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)
        fold_accuracies = []
        
        for step, (train_index, val_index) in enumerate(tscv.split(X_train)):
            X_t_cv = X_train.iloc[train_index].copy()
            X_v_cv = X_train.iloc[val_index].copy()
            y_t_cv = y_train.iloc[train_index].copy()
            y_v_cv = y_train.iloc[val_index].copy()
            
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
                
            clf_cv = FastHMMClassifier(**params) 
            clf_cv.fit(X_t_processed, y_t_cv)
            val_preds = clf_cv.predict(X_v_processed)
            
            current_acc = accuracy_score(y_v_cv, val_preds)
            fold_accuracies.append(current_acc)
            
            trial.report(np.mean(fold_accuracies), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
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


def run_hmm_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, add_pca=False, validation="holdout", n_splits=5, tscv_test_size=None):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "hmm_model.joblib"
    scaler_path = output_dir / "hmm_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training and inferring immediately!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print("Scaling features for HMM...")
    
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
        print("Applying PCA for HMM (Retaining 95% variance)...")
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        pca_path = output_dir / "hmm_pca.joblib"
        joblib.dump(pca, pca_path)
        
        if X_val_scaled is not None:
            X_val_processed = pca.transform(X_val_scaled)
    else:
        X_train_processed = X_train_scaled
        if X_val_scaled is not None:
            X_val_processed = X_val_scaled

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "hmm_optuna.db"
    storage_url = f"sqlite:///{db_path.absolute()}"
    
    print(f"\nStarting Optuna search ({n_trials} trials | HMM)...")
    study = optuna.create_study(
        study_name="hmm_optimization",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, add_pca, validation, n_splits, tscv_test_size),
        n_trials=n_trials,
        n_jobs=-1,
        catch=(Exception,)
    )
    
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    print("Training final model with optimal parameters...")
    final_clf = FastHMMClassifier(**best_params, random_state=42)
    final_clf.fit(X_train_processed, y_train)
    
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
        final_feature_names = list(X_train_features.columns)

    config = {
        "model_type": "HMM",
        "best_params": best_params,
        "train_accuracy": train_acc,
        "val_accuracy": study.best_value,
        "pca_applied": add_pca,
        "features_used": final_feature_names
    }
    with open(output_dir / "hmm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_optuna_history(study, reports_dir)
    
    return final_clf, scaler
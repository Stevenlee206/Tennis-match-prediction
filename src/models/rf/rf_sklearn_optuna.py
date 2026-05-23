import json
import joblib
from joblib import parallel_backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # <--- ADDED PCA IMPORT
from sklearn.metrics import accuracy_score

# =====================================================================
# HOTFIX: Patch Scikit-Learn (Kept from original)
# =====================================================================
import sklearn.ensemble._forest as forest
_original_forest_init = forest.ForestClassifier.__init__

def _patched_forest_init(self, *args, **kwargs):
    if 'base_estimator' in kwargs:
        kwargs['estimator'] = kwargs.pop('base_estimator')
    _original_forest_init(self, *args, **kwargs)

forest.ForestClassifier.__init__ = _patched_forest_init
# =====================================================================

# Try to import RotationForest (Kept from original)
try:
    from rotation_forest import RotationForestClassifier
    import rotation_forest.rotation_forest as rf_module
    from rotation_forest.rotation_forest import RotationTreeClassifier
    from sklearn.exceptions import NotFittedError
    from sklearn.tree import DecisionTreeClassifier

    rf_module.NotFittedError = NotFittedError

    def _patched_reduce(self):
        res = DecisionTreeClassifier.__reduce__(self)
        if len(res) == 3:
            func, args, state = res
            if isinstance(state, dict):
                for attr in ['rotation_matrix_', 'pca_']:
                    if hasattr(self, attr):
                        state[attr] = getattr(self, attr)
            return (func, args, state)
        return res

    def _patched_setstate(self, state):
        custom_attrs = {attr: state.pop(attr) for attr in ['rotation_matrix_', 'pca_'] if attr in state}
        DecisionTreeClassifier.__setstate__(self, state)
        for attr, val in custom_attrs.items():
            setattr(self, attr, val)

    RotationTreeClassifier.__reduce__ = _patched_reduce
    RotationTreeClassifier.__setstate__ = _patched_setstate

    HAS_ROT_FOREST = True
except ImportError:
    HAS_ROT_FOREST = False
try:
    from sktree import ObliqueRandomForestClassifier
    HAS_OBLIQUE = True
except ImportError:
    HAS_OBLIQUE = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble._forest import _generate_sample_indices

class WeightedRandomForestClassifier(RandomForestClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tree_weights_ = None

    def fit(self, X, y, sample_weight=None):
        # 1. Fit the standard Random Forest normally
        super().fit(X, y, sample_weight=sample_weight)

        # 2. Grade each tree using its Out-Of-Bag (OOB) samples
        n_samples = X.shape[0]
        self.tree_weights_ = np.zeros(self.n_estimators)

        # Ensure X and y are arrays for safe indexing
        X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y

        for i, tree in enumerate(self.estimators_):
            # Find the indices this specific tree used for training
            train_indices = _generate_sample_indices(tree.random_state, n_samples, n_samples)
            
            # The OOB samples are everything NOT in the train_indices
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[train_indices] = False

            if not np.any(oob_mask):
                self.tree_weights_[i] = 1.0 # Fallback if no OOB data exists
                continue

            X_oob = X_arr[oob_mask]
            y_oob = y_arr[oob_mask]

            # Grade the tree (Applying your custom upset weights if they exist!)
            if sample_weight is not None:
                w_oob = sample_weight[oob_mask]
                oob_acc = accuracy_score(y_oob, tree.predict(X_oob), sample_weight=w_oob)
            else:
                oob_acc = accuracy_score(y_oob, tree.predict(X_oob))

            # Penalize trees worse than a coin flip. 
            # We subtract 0.5 so a 50% accurate tree gets weight near 0.
            self.tree_weights_[i] = max(0.0001, oob_acc - 0.50)

        # 3. Normalize the weights so they sum to 1.0
        self.tree_weights_ = self.tree_weights_ / np.sum(self.tree_weights_)
        return self

    def predict_proba(self, X):
        # Get raw probabilities from all trees. Shape: (n_estimators, n_samples, n_classes)
        all_probas = np.array([tree.predict_proba(X) for tree in self.estimators_])

        # Calculate the weighted sum of probabilities using tensor dot product
        # This replaces the standard mean() average
        weighted_probas = np.tensordot(self.tree_weights_, all_probas, axes=(0, 0))
        return weighted_probas

    def predict(self, X):
        # Determine the final class based on highest weighted probability
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
def generate_sample_weights(X_raw, y_raw, strategy="none", base_weight=1.0):
    """
    Dynamically routes and calculates sample weights based on the chosen strategy.
    """
    n_samples = len(y_raw)
    weights = np.ones(n_samples)
    
    if strategy == "none" or base_weight <= 1.0 or 'elo_diff' not in X_raw.columns:
        return weights

    # Ensure y is a numpy array for safe boolean indexing
    y_vals = y_raw.values if isinstance(y_raw, pd.Series) else y_raw
    elo_diffs = X_raw['elo_diff'].values
    
    # Base Upset Mask: P1 has higher Elo but loses (y==0) OR P1 has lower Elo but wins (y==1)
    upset_mask = ((elo_diffs > 0) & (y_vals == 0)) | ((elo_diffs < 0) & (y_vals == 1))

    # ---------------------------------------------------------
    # STRATEGY 1: STATIC (The Hardcoded Flag)
    # Every upset gets the exact same flat penalty.
    # ---------------------------------------------------------
    if strategy == "static":
        weights[upset_mask] = base_weight

    # ---------------------------------------------------------
    # STRATEGY 2: MAGNITUDE (Proportional to the Surprise)
    # A 300 Elo upset penalizes the model far more than a 10 Elo upset.
    # ---------------------------------------------------------
    elif strategy == "magnitude":
        for i in range(n_samples):
            if upset_mask[i]:
                # Calculate how many "100-point Elo gaps" were overcome
                gap_severity = abs(elo_diffs[i]) / 100.0
                # Scale the weight dynamically. 
                # (e.g., A 200 Elo upset with a base_weight of 2.0 = 1.0 + (2.0 * 2) = Weight of 5.0)
                weights[i] = 1.0 + (base_weight * gap_severity)

    # ---------------------------------------------------------
    # STRATEGY 3: TEMPORAL (Recency Decay)
    # Upsets from 10 years ago matter less than upsets from last month.
    # Assumes data is sorted chronologically.
    # ---------------------------------------------------------
    elif strategy == "temporal":
        # Create an exponential curve from near 0.0 (oldest match) to 1.0 (newest match)
        decay_curve = np.exp(np.linspace(-3, 0, n_samples))
        
        for i in range(n_samples):
            if upset_mask[i]:
                # Old upsets get a weight close to 1.0. New upsets get the full base_weight.
                weights[i] = 1.0 + ((base_weight - 1.0) * decay_curve[i])

    return weights
# ---> ADDED add_pca ARGUMENT
def objective(trial, X_train, y_train, X_val, y_val, n_est_min, n_est_max, depth_min, depth_max, variant="rf", add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0, n_splits=5, tscv_test_size=None):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", n_est_min, n_est_max),
        "max_depth": trial.suggest_int("max_depth", depth_min, depth_max),
        
        # INCREASED: Force at least 15-50 matches before allowing a split
        "min_samples_split": trial.suggest_int("min_samples_split", 15, 50),
        
        # ADDED: Only use a random subset of features per split (forces diversity)
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        
        "random_state": 42,
    }
    
    if variant == "extra_trees":
        params["n_jobs"] = -1
        clf = ExtraTreesClassifier(**params)
    elif variant == "rrf":
        params["ccp_alpha"] = trial.suggest_float("ccp_alpha", 0.0, 0.05)
        params["n_jobs"] = -1
        clf = RandomForestClassifier(**params)
    elif variant == "rotation_forest":
        clf = RotationForestClassifier(**params)
    elif variant == "oblique":
        if not HAS_OBLIQUE:
            raise ImportError("Please run: pip install sktree")
        # Tune how many features are combined per linear projection (default is usually 1.5)
        params["feature_combinations"] = trial.suggest_float("feature_combinations", 1.2, 5.0)
        params["n_jobs"] = -1
        clf = ObliqueRandomForestClassifier(**params)
    elif variant == "weighted":
        params["n_jobs"] = -1
        # It takes standard RF parameters, so Optuna tunes it exactly like a normal RF
        clf = WeightedRandomForestClassifier(**params)
    else: 
        params["n_jobs"] = -1
        clf = RandomForestClassifier(**params)
        
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
        
        with parallel_backend("threading"):
            if variant == "rotation_forest":
                clf.fit(X_t_processed, y_train)
            else:
                clf.fit(X_t_processed, y_train, sample_weight=weights)
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
                
            # ---> GENERATE AND APPLY WEIGHTS FOR THIS FOLD
            weights_cv = generate_sample_weights(X_t_cv, y_t_cv, weight_strategy, upset_weight)
            
            with parallel_backend("threading"):
                if variant == "rotation_forest":
                    clf.fit(X_t_processed, y_t_cv)
                else:
                    clf.fit(X_t_processed, y_t_cv, sample_weight=weights_cv)
                val_preds = clf.predict(X_v_processed)
            
            fold_accuracies.append(accuracy_score(y_v_cv, val_preds))
            
        return np.mean(fold_accuracies)

# (plot_optuna_history stays exactly the same)
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

# (plot_feature_importance stays mostly the same)
def plot_feature_importance(clf, feature_names, save_path, variant):
    if variant in ["rotation_forest", "oblique"]:
        print(f"Skipping feature importance plot (Not natively supported by {variant.upper()}).")
        return

    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis")
    plt.title(f"{variant.upper()} Feature Importance (Gini Impurity Decrease)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

# ---> ADDED add_pca ARGUMENT
def run_rf_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, 
                    n_trials=30, n_est_min=50, n_est_max=500, depth_min=5, depth_max=50, 
                    variant="rf", add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0, n_splits=3, tscv_test_size=None):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{variant}_model.joblib"
    scaler_path = output_dir / f"{variant}_scaler.joblib"

    print(f"Scaling features for {variant.upper()}...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 1. Safely handle Walk-Forward 'None'
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    if add_pca:
        print(f"Applying PCA for {variant.upper()} (Retaining 95% variance)...")
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        pca_path = output_dir / f"{variant}_pca.joblib"
        joblib.dump(pca, pca_path)
        
        # 2. Keep Validation Set mathematically aligned if it exists
        if X_val_scaled is not None:
            X_val_processed = pca.transform(X_val_scaled)
    else:
        X_train_processed = X_train_scaled
        if X_val_scaled is not None:
            X_val_processed = X_val_scaled

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / f"{variant}_sklearn_optuna.db"
    study_name = f"{variant}_optimization_{output_dir.name}"
    
    print(f"\nStarting Optuna hyperparameter search ({n_trials} trials for {variant.upper()} | Upset Weight: {upset_weight}x)...")
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path.absolute()}",
        load_if_exists=True,
        direction="maximize"
    )
    
    # ---> PASSED upset_weight TO THE OBJECTIVE FUNCTION
    print(f"\nStarting Optuna search ({n_trials} trials | Variant: {variant.upper()} | Strategy: {weight_strategy.upper()} | Base Wt: {upset_weight})...")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, n_est_min, n_est_max, depth_min, depth_max, variant, add_pca, validation, weight_strategy, upset_weight, n_splits, tscv_test_size),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    if variant == "extra_trees":
        final_clf = ExtraTreesClassifier(**best_params, random_state=42, n_jobs=-1)
    elif variant == "rrf":
        final_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif variant == "rotation_forest":
        final_clf = RotationForestClassifier(**best_params, random_state=42)
    elif variant == "oblique":
        final_clf = ObliqueRandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif variant == "weighted":
        # It takes standard RF parameters, so Optuna tunes it exactly like a normal RF
        final_clf = WeightedRandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    else:
        final_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        
    final_weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
    
    with parallel_backend("threading"):
        if variant == "rotation_forest":
            final_clf.fit(X_train_processed, y_train)
        else:
            final_clf.fit(X_train_processed, y_train, sample_weight=final_weights)    
    # ==========================================
    # ---> ADDED: OVERFITTING DIAGNOSTIC <---
    # ==========================================
    train_preds = final_clf.predict(X_train_processed)
    train_acc = accuracy_score(y_train, train_preds)
    
    print("\n" + "-"*30)
    print(" OVERFITTING CHECK")
    print("-"*30)
    print(f"Training Accuracy:     {train_acc * 100:.2f}%")
    print(f"Optuna Val Accuracy:   {study.best_value * 100:.2f}%")
    
    # If the model scores >10% better on the training set, it is memorizing data
    if (train_acc - study.best_value) > 0.10:
        print("⚠️ WARNING: High likelihood of overfitting. The model is memorizing the training data.")
        print("💡 TIP: Try lowering --rf_depth_max or increasing min_samples_split.")
    print("-" * 30 + "\n")  
    joblib.dump(final_clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    # ---> DYNAMIC FEATURE NAMES IF PCA IS ENABLED
    if add_pca:
        final_feature_names = [f"PC{i+1}" for i in range(X_train_processed.shape[1])]
    else:
        final_feature_names = list(X_train.columns)

    config = {
        "model_type": variant.upper(),
        "optimizer": "Optuna",
        "best_params": best_params,
        "train_accuracy": train_acc,
        "val_accuracy": study.best_value,
        "pca_applied": add_pca,
        "weight_strategy": weight_strategy, # <--- NEW
        "upset_weight": upset_weight,       # <--- NEW
        "features_used": final_feature_names
    }
    with open(output_dir / f"{variant}_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Generating plots...")
    plot_optuna_history(study, reports_dir)
    plot_feature_importance(final_clf, final_feature_names, reports_dir, variant)
    
    return final_clf, scaler
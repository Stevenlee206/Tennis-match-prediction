from copy import deepcopy
from math import ceil
from pathlib import Path
import json
import joblib
import numpy as np
import optuna
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from optuna.exceptions import TrialPruned
from functools import partial

from src.config.hmm_config import (
    HMM_DEFAULT_N_TRIALS,
    HMM_EARLY_STOP_PATIENCE,
    HMM_MAX_EPOCHS,
    HMM_PRUNER_STARTUP_TRIALS,
    HMM_TUNING,
)
from src.models.hmm.hmm_model import HMMClassifier
from src.utils.metrics import accuracy_from_predictions
from src.utils.paths import ensure_writable_path
from src.utils.plots import plot_optuna_history

def _prepare_features(X_train, X_val):
    if "is_augmented" in X_train.columns:
        X_train_features = X_train.drop(columns=["is_augmented"])
    else:
        X_train_features = X_train

    if X_val is not None:
        X_val_features = X_val.drop(columns=["is_augmented"], errors="ignore")
    else:
        X_val_features = None

    return X_train_features, X_val_features


def _train_with_early_stopping(
    clf,
    X_train_processed,
    y_train,
    X_val_processed,
    y_val,
    patience=HMM_EARLY_STOP_PATIENCE,
    max_epochs=HMM_MAX_EPOCHS,
    sequence_length=None,
):
    best_score = -np.inf
    best_epoch = 0
    best_models = None
    epochs_without_improvement = 0
    current_models = None
    best_models_tmpfile = None

    for epoch in range(1, max_epochs + 1):
        classes, majority_class, current_models = clf._train_class_models(
            X_train_processed,
            y_train,
            n_iter=1,
            warm_start_models=current_models,
        )
        clf.classes_ = classes
        clf.majority_class_ = majority_class
        clf.models_ = current_models

        # Limit context passed to predict to reduce memory usage
        try:
            ctx = None
            if sequence_length is not None and X_train_processed is not None:
                try:
                    ctx_rows = int(sequence_length)
                    if getattr(X_train_processed, 'shape', None):
                        ctx = X_train_processed[-ctx_rows:]
                except Exception:
                    ctx = None

            val_preds = clf.predict(X_val_processed, context=ctx)
            val_score = accuracy_from_predictions(y_val, val_preds)
        except Exception:
            val_score = 0.0

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            # snapshot models to disk to avoid keeping two large copies in RAM
            try:
                import tempfile, os
                if best_models_tmpfile:
                    try:
                        best_models_tmpfile.close()
                    except Exception:
                        pass
                best_models_tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.joblib')
                joblib.dump(current_models, best_models_tmpfile.name)
                best_models = best_models_tmpfile.name
            except Exception:
                best_models = deepcopy(current_models)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # restore best models if they were saved to temp file
    if best_models is not None:
        try:
            if isinstance(best_models, str):
                clf.models_ = joblib.load(best_models)
                try:
                    import os
                    os.unlink(best_models)
                except Exception:
                    pass
            else:
                clf.models_ = best_models
        except Exception:
            clf.models_ = best_models

    try:
        if best_models_tmpfile:
            best_models_tmpfile.close()
    except Exception:
        pass

    return float(best_score), int(best_epoch), clf


def _fit_and_score(
    X_train,
    y_train,
    X_val,
    y_val,
    n_components,
    covariance_type,
    tol,
    min_covar,
    sequence_length,
    add_pca=False,
    patience=HMM_EARLY_STOP_PATIENCE,
    max_epochs=HMM_MAX_EPOCHS,
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Cast to float32 to reduce memory footprint (most sklearn/hmmlearn accept float32)
    X_train_scaled = X_train_scaled.astype("float32")
    X_val_scaled = X_val_scaled.astype("float32")

    if add_pca:
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        X_val_processed = pca.transform(X_val_scaled)
    else:
        X_train_processed = X_train_scaled
        X_val_processed = X_val_scaled

    clf = HMMClassifier(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=max_epochs,
        tol=tol,
        min_covar=min_covar,
        sequence_length=sequence_length,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        val_score, best_epoch, clf = _train_with_early_stopping(
            clf,
            X_train_processed,
            y_train,
            X_val_processed,
            y_val,
            patience=patience,
            max_epochs=max_epochs,
            sequence_length=sequence_length,
        )
    # free large arrays no longer needed in caller
    try:
        del X_train_scaled
        del X_val_scaled
    except Exception:
        pass
    import gc
    gc.collect()

    return val_score, best_epoch, clf, scaler


def objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    n_components_min=HMM_TUNING.n_components_min,
    n_components_max=HMM_TUNING.n_components_max,
    seq_len_min=HMM_TUNING.seq_len_min,
    seq_len_max=HMM_TUNING.seq_len_max,
    validation="walk_forward",
    add_pca=False,
    n_splits=5,
    tscv_test_size=None,
    max_epochs=HMM_MAX_EPOCHS,
):
    print(f"[Optuna] Trial {trial.number + 1} started", flush=True)
    n_components = trial.suggest_int("n_components", n_components_min, n_components_max)
    covariance_type = trial.suggest_categorical("covariance_type", list(HMM_TUNING.covariance_types))
    tol = trial.suggest_float("tol", HMM_TUNING.tol_min, HMM_TUNING.tol_max, log=True)
    min_covar = trial.suggest_float("min_covar", HMM_TUNING.min_covar_min, HMM_TUNING.min_covar_max, log=True)
    sequence_length = trial.suggest_int("sequence_length", seq_len_min, seq_len_max)

    X_train_features, X_val_features = _prepare_features(X_train, X_val)

    if validation == "holdout":
        try:
            score, best_epoch, _, _ = _fit_and_score(
                X_train_features,
                y_train,
                X_val_features,
                y_val,
                n_components,
                covariance_type,
                tol,
                min_covar,
                sequence_length,
                add_pca=add_pca,
                max_epochs=max_epochs,
            )
            trial.set_user_attr("fold_best_epochs", [best_epoch])
            return float(score)
        except Exception:
            trial.set_user_attr("fold_best_epochs", [1])
            return 0.0

    if validation == "walk_forward":
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)
        fold_scores = []
        fold_best_epochs = []

        for fold_idx, (train_index, val_index) in enumerate(tscv.split(X_train_features), start=1):
            X_t_cv = X_train_features.iloc[train_index].copy()
            X_v_cv = X_train_features.iloc[val_index].copy()
            y_t_cv = y_train.iloc[train_index].copy()
            y_v_cv = y_train.iloc[val_index].copy()

            if len(np.unique(y_t_cv)) < 2 or len(y_v_cv) == 0:
                continue

            try:
                score, best_epoch, _, _ = _fit_and_score(
                    X_t_cv,
                    y_t_cv,
                    X_v_cv,
                    y_v_cv,
                    n_components,
                    covariance_type,
                    tol,
                    min_covar,
                    sequence_length,
                    add_pca=add_pca,
                    max_epochs=max_epochs,
                )
                fold_scores.append(score)
                fold_best_epochs.append(best_epoch)
                trial.report(float(np.mean(fold_scores)), step=fold_idx)
                if trial.should_prune():
                    raise TrialPruned(f"Pruned at fold {fold_idx}")
            except TrialPruned:
                raise
            except Exception:
                fold_scores.append(0.0)
                fold_best_epochs.append(1)
                trial.report(float(np.mean(fold_scores)), step=fold_idx)
                if trial.should_prune():
                    raise TrialPruned(f"Pruned at fold {fold_idx}")

        if not fold_scores:
            trial.set_user_attr("fold_best_epochs", [1])
            return 0.0

        trial.set_user_attr("fold_best_epochs", fold_best_epochs or [1])
        return float(np.mean(fold_scores))

    raise ValueError(f"Unsupported validation mode: {validation}")


def run_hmm_hpo(
    X_train,
    y_train,
    X_val,
    y_val,
    output_dir,
    reports_dir,
    n_trials=HMM_DEFAULT_N_TRIALS,
    n_jobs=1,
    n_components_min=HMM_TUNING.n_components_min,
    n_components_max=HMM_TUNING.n_components_max,
    n_iter_min=HMM_TUNING.n_iter_min,
    n_iter_max=HMM_TUNING.n_iter_max,
    seq_len_min=HMM_TUNING.seq_len_min,
    seq_len_max=HMM_TUNING.seq_len_max,
    add_pca=False,
    validation="holdout",
    n_splits=5,
    tscv_test_size=None,
    max_epochs=HMM_TUNING.n_iter_max,
    resume=False,
):
    output_dir = ensure_writable_path(output_dir)
    reports_dir = ensure_writable_path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "hmm_model.joblib"
    scaler_path = output_dir / "hmm_scaler.joblib"

    if model_path.exists() and scaler_path.exists() and not resume:
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training and inferring immediately!")
        return joblib.load(model_path), joblib.load(scaler_path)

    print("Scaling features for HMM...")
    X_train_features, X_val_features = _prepare_features(X_train, X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features) if X_val_features is not None else None

    # Cast to float32 to reduce memory footprint
    X_train_scaled = X_train_scaled.astype("float32")
    if X_val_scaled is not None:
        X_val_scaled = X_val_scaled.astype("float32")

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
    # free scaled copies if PCA produced new arrays
    try:
        del X_train_scaled
        del X_val_scaled
    except Exception:
        pass
    import gc
    gc.collect()

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "hmm_optuna.db"
    storage_url = f"sqlite:///{db_path.absolute()}"

    # NOTE: previously this code forced `n_jobs=1` when using SQLite storage
    # due to DB locking and pickling restrictions. That forced behavior was
    # removed to allow users to control parallelism explicitly.

    print(
        f"\nStarting Optuna search ({n_trials} trials | HMM | states: [{n_components_min}, {n_components_max}] | seq_len: [{seq_len_min}, {seq_len_max}] | early_stop: patience={HMM_EARLY_STOP_PATIENCE}, max_epochs={max_epochs})..."
    )

    def log_trial(study, trial):
        if trial.value is None:
            return
        print(
            f"[Optuna] Trial {trial.number + 1}/{n_trials} finished | value={trial.value:.4f} | best={study.best_value:.4f}"
        )

    study = optuna.create_study(
        study_name="hmm_optimization",
        storage=storage_url,
        load_if_exists=resume,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=HMM_PRUNER_STARTUP_TRIALS),
        direction="maximize",
    )
    # Build a picklable partial objective so Optuna can spawn workers when needed
    obj_func = partial(
        objective,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_components_min=n_components_min,
        n_components_max=n_components_max,
        seq_len_min=seq_len_min,
        seq_len_max=seq_len_max,
        validation=validation,
        add_pca=add_pca,
        n_splits=n_splits,
        tscv_test_size=tscv_test_size,
        max_epochs=max_epochs,
    )

    study.optimize(
        obj_func,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[log_trial],
        gc_after_trial=True,
    )

    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    # Recover fold epoch info robustly when resuming
    best_trial_epochs = study.best_trial.user_attrs.get("fold_best_epochs") if study.best_trial is not None else None

    if not best_trial_epochs:
        # Try to find the highest-valued trial that recorded fold_best_epochs
        trials_sorted = sorted(
            [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)],
            key=lambda t: (t.value if t.value is not None else -np.inf),
            reverse=True,
        )
        found = False
        for t in trials_sorted:
            attrs = t.user_attrs.get("fold_best_epochs")
            if attrs:
                best_trial_epochs = attrs
                found = True
                print(f"[Optuna] Using fold_best_epochs from trial {t.number} (value={t.value}) as fallback for median computation.")
                break

        if not found:
            best_trial_epochs = [1]

    median_best_epoch = max(1, int(ceil(float(np.median(best_trial_epochs)))))
    print(f"Best epochs per fold: {best_trial_epochs} | Median epoch: {median_best_epoch}")

    final_model = HMMClassifier(
        n_components=best_params["n_components"],
        covariance_type=best_params["covariance_type"],
        n_iter=median_best_epoch,
        tol=best_params["tol"],
        min_covar=best_params["min_covar"],
        sequence_length=best_params["sequence_length"],
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model.fit(X_train_processed, y_train)

    train_preds = final_model.predict(X_train_processed)
    train_acc = accuracy_from_predictions(y_train, train_preds)

    print("\n" + "-" * 30)
    print(" OVERFITTING CHECK")
    print("-" * 30)
    print(f"Training Accuracy:     {train_acc * 100:.2f}%")
    print(f"Optuna Val Accuracy:   {study.best_value * 100:.2f}%")
    if (train_acc - study.best_value) > 0.10:
        print("WARNING: High likelihood of overfitting. The model is memorizing the training data.")
    print("-" * 30 + "\n")

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)

    if add_pca:
        n_features = X_train_processed.shape[1]
        final_feature_names = [f"PC{i + 1}" for i in range(n_features)]
    else:
        final_feature_names = list(X_train_features.columns)

    config = {
        "model_type": "HMM",
        "best_params": best_params,
        "fold_best_epochs": best_trial_epochs,
        "median_best_epoch": median_best_epoch,
        "train_accuracy": train_acc,
        "val_accuracy": study.best_value,
        "pca_applied": add_pca,
        "sequence_length": best_params["sequence_length"],
        "features_used": final_feature_names,
        "early_stopping_patience": HMM_EARLY_STOP_PATIENCE,
        "max_epochs": max_epochs,
    }

    with open(output_dir / "hmm_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("Generating plots...")
    plot_optuna_history(study, reports_dir)

    return final_model, scaler

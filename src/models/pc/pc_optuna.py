import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score
from importlib import import_module

import torch
import torch.nn.functional as F

from src.models.preco.pc_network_torch import PCNetworkConfig
from src.models.preco.pc_network_torch import PredictiveCodingNetworkTorch
from src.models.utils.metrics import binary_classification_metrics

# Re-use bias metric and weight generation
def generate_sample_weights(X_train, y_train, weight_strategy="none", upset_weight=1.0):
    if weight_strategy == "none":
        return None
    weights = np.ones(len(y_train))
    # Simple static strategy fallback
    if weight_strategy == "static" and 'elo_diff' in X_train.columns:
        upsets = (X_train['elo_diff'] > 0) != y_train
        weights[upsets] = upset_weight
    return weights

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

def plot_feature_importance_permutation(model, X_val, y_val, feature_names, save_path):
    # Permutation feature importance for PC
    baseline_probs = model.predict_proba(X_val)
    baseline_preds = (baseline_probs >= 0.5).astype(int)
    baseline_acc = accuracy_score(y_val, baseline_preds)

    importances = []
    n_features = X_val.shape[1]
    
    for i in range(n_features):
        X_val_shuffled = X_val.copy()
        np.random.shuffle(X_val_shuffled[:, i])
        probs = model.predict_proba(X_val_shuffled)
        preds = (probs >= 0.5).astype(int)
        shuffled_acc = accuracy_score(y_val, preds)
        importances.append(baseline_acc - shuffled_acc)
        
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette="viridis", legend=False)
    plt.title("Predictive Coding Permutation Feature Importance")
    plt.xlabel("Accuracy Drop")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

def plot_training_curve(metrics_hist, save_path):
    epochs = [m["epoch"] for m in metrics_hist]
    val_accs = [m["val_acc"] for m in metrics_hist]
    train_accs = [m["train_acc"] for m in metrics_hist]
    train_energies = [m["train_energy"] for m in metrics_hist]
    train_losses = [m.get("train_loss", 0.0) for m in metrics_hist]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Loss & Energy
    ax1.plot(epochs, train_energies, color='tab:red', label='Train Energy')
    ax1.plot(epochs, train_losses, color='tab:orange', label='Train BCE Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss / Energy')
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # Plot 2: Accuracies
    ax2.plot(epochs, train_accs, color='tab:blue', label='Train Accuracy')
    ax2.plot(epochs, val_accs, color='tab:green', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)
    
    fig.tight_layout()  
    plt.suptitle("Predictive Coding Training Curve", y=1.05)
    plt.savefig(save_path / "training_curve.png", dpi=300)
    plt.close()

def train_pc_model_loop(model, X_t, y_t, X_v=None, y_v=None, weights_t=None, max_epochs=100, batch_size=256, verbose=False, patience=10):
    t0 = time.perf_counter()
    rng = np.random.default_rng(model.cfg.random_seed)
    
    best_acc = 0.0
    best_state = None
    best_epoch = max_epochs
    metrics_history = []
    epochs_no_improve = 0
    
    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    X_t_t = torch.tensor(X_t, device=device, dtype=torch.float32)
    y_t_t = torch.tensor(y_t, device=device, dtype=torch.float32).view(-1)
    
    if X_v is not None and y_v is not None:
        X_v_t = torch.tensor(X_v, device=device, dtype=torch.float32)
        y_v_t = torch.tensor(y_v, device=device, dtype=torch.float32).view(-1)
    else:
        X_v_t, y_v_t = None, None

    w_t = None
    if weights_t is not None:
        w_t = torch.tensor(weights_t, device=device, dtype=torch.float32).view(-1)

    n_samples = X_t_t.shape[0]
    indices = np.arange(n_samples)
    
    for epoch in range(1, max_epochs + 1):
        rng.shuffle(indices)
        energies = []
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_idx_t = torch.tensor(batch_idx, device=device, dtype=torch.long)
            xb = X_t_t.index_select(0, batch_idx_t)
            yb = y_t_t.index_select(0, batch_idx_t).view(-1, 1)
            
            w_opt = None
            if weights_t is not None:
                w_opt = w_t.index_select(0, batch_idx_t)
            
            energy = model.train_on_batch(xb, yb, sample_weights=w_opt)
            energies.append(energy)
            
        # Eval on train (device)
        train_probs_t = model.predict_proba_torch(X_t_t)
        train_preds_t = (train_probs_t >= 0.5).to(torch.float32)
        train_acc = float((train_preds_t == y_t_t).to(torch.float32).mean().item())

        # Explicit train loss (BCE)
        train_loss = float(F.binary_cross_entropy(train_probs_t, y_t_t).item())
        
        avg_energy = np.mean(energies)
        
        if X_v_t is not None:
            # Eval on val (device)
            val_probs_t = model.predict_proba_torch(X_v_t)
            val_preds_t = (val_probs_t >= 0.5).to(torch.float32)
            acc = float((val_preds_t == y_v_t).to(torch.float32).mean().item())
        else:
            acc = None
            
        metrics_history.append({
            "epoch": epoch,
            "train_energy": avg_energy,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": acc
        })
        
        if verbose:
            if acc is not None:
                print(f"Epoch {epoch:03d}/{max_epochs} | Train Energy: {avg_energy:.4f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f}")
            else:
                print(f"Epoch {epoch:03d}/{max_epochs} | Train Energy: {avg_energy:.4f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            
        # Use acc for early stopping if available, otherwise just use the final epoch state
        current_acc = acc if acc is not None else train_acc
        
        if current_acc >= best_acc:
            best_acc = current_acc
            best_epoch = epoch
            best_state = model.get_state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if X_v_t is not None and epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
            
    if best_state is None:
        best_state = model.get_state_dict()
            
    train_time = time.perf_counter() - t0
    if verbose:
        print(f"\n=> Trained {max_epochs} epochs in {train_time:.1f}s. Best Val Acc: {best_acc:.4f} at epoch {best_epoch}\n")
        
    return best_acc, best_epoch, best_state, metrics_history, train_time

def objective(trial, X_train, y_train, X_val, y_val, add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0, precomputed_folds=None):
    # Hyperparams to tune
    cfg_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "inference_lr": trial.suggest_float("inference_lr", 1e-3, 1.0, log=True),
        "inference_steps": trial.suggest_int("inference_steps", 5, 50),
        "hidden_activation": trial.suggest_categorical("hidden_activation", ["tanh", "relu"]),
        "output_activation": "sigmoid",
        "random_seed": 42
    }
    
    # Depth and Width
    depth = trial.suggest_int("depth", 1, 3)
    width = trial.suggest_categorical("width", [16, 32, 64, 128])
    hidden_sizes = [width] * depth
    
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    epochs = trial.suggest_categorical("epochs", [20, 50, 100])
    
    if validation == "holdout":
        v_drop_cols = ['is_augmented', 'winner_name', 'loser_name', 'match_id'] 
        v_drop_cols = [c for c in v_drop_cols if c in X_val.columns]
        t_drop_cols = ['is_augmented', 'winner_name', 'loser_name', 'match_id']
        t_drop_cols = [c for c in t_drop_cols if c in X_train.columns]
        
        X_t_clean = X_train.drop(columns=t_drop_cols, errors='ignore')
        X_v_clean = X_val.drop(columns=v_drop_cols, errors='ignore')

        scaler = StandardScaler()
        X_t_scaled = scaler.fit_transform(X_t_clean).astype(np.float32)
        X_v_scaled = scaler.transform(X_v_clean).astype(np.float32)
        
        y_t_np = y_train.values.astype(np.float32)
        y_v_np = y_val.values.astype(np.float32)
        
        weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
        
        pc_cfg = PCNetworkConfig(**cfg_params)
        layer_sizes = [X_t_scaled.shape[1], *hidden_sizes, 1]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=pc_cfg, device=device)
        
        best_acc, _, _, _, _ = train_pc_model_loop(model, X_t_scaled, y_t_np, X_v_scaled, y_v_np, weights, max_epochs=epochs, batch_size=batch_size, verbose=False)
        return best_acc
        
    elif validation == "walk_forward":
        fold_accuracies = []
        
        for train_df, val_df in precomputed_folds:
            y_t_cv = train_df['target']
            X_t_cv = train_df.drop(columns=['target', 'year'], errors='ignore')
            
            y_v_cv = val_df['target']
            X_v_cv = val_df.drop(columns=['target', 'year'], errors='ignore')
            
            # Keep only non-augmented data for validation
            if 'is_augmented' in X_v_cv.columns:
                y_v_cv = y_v_cv[X_v_cv['is_augmented'] == 0]
                X_v_cv = X_v_cv[X_v_cv['is_augmented'] == 0]
            
            v_drop_cols = ['is_augmented', 'winner_name', 'loser_name', 'match_id']
            v_drop_cols = [c for c in v_drop_cols if c in X_v_cv.columns]
            t_drop_cols = ['is_augmented', 'winner_name', 'loser_name', 'match_id']
            t_drop_cols = [c for c in t_drop_cols if c in X_t_cv.columns]
            
            X_v_clean = X_v_cv.drop(columns=v_drop_cols, errors='ignore')
            X_t_clean = X_t_cv.drop(columns=t_drop_cols, errors='ignore')
            
            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t_clean).astype(np.float32)
            X_v_scaled = scaler.transform(X_v_clean).astype(np.float32)
            
            weights_cv = generate_sample_weights(X_t_cv, y_t_cv, weight_strategy, upset_weight)
            
            pc_cfg = PCNetworkConfig(**cfg_params)
            layer_sizes = [X_t_scaled.shape[1], *hidden_sizes, 1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=pc_cfg, device=device)
            
            y_t_np = y_t_cv.values.astype(np.float32)
            y_v_np = y_v_cv.values.astype(np.float32)
            
            acc, _, _, _, _ = train_pc_model_loop(model, X_t_scaled, y_t_np, X_v_scaled, y_v_np, weights_cv, max_epochs=epochs, batch_size=batch_size, verbose=False)
            fold_accuracies.append(acc)
            
        return np.mean(fold_accuracies)

def run_pc_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, 
                    n_trials=30, validation="holdout", optimizer="optuna",
                    weight_strategy="none", upset_weight=1.0, precomputed_folds=None):
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "pc_model.pt"
    scaler_path = output_dir / "pc_scaler.joblib"
    config_path = output_dir / "pc_config.json"
    log_path = output_dir / "log.txt"

    print(f"\nStarting Predictive Coding Tuning pipeline...")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "pc_optuna.db"
    storage_url = f"sqlite:///{db_path.absolute()}"
    
    study = optuna.create_study(
        study_name="pc_optimization",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"Optimizing Hyperparams with {optimizer.upper()} for {n_trials} trials...")
    if optimizer == "optuna":
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, False, validation, weight_strategy, upset_weight, precomputed_folds),
            n_trials=n_trials
        )
    else: 
        # Grid Search logic handled by optuna sampler setup mapping if needed, but for simplicity we rely on Optuna TPESampler.
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, False, validation, weight_strategy, upset_weight, precomputed_folds),
            n_trials=n_trials
        )
        
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    # Retrain Phase
    print("Retraining final model on full train set with optimal parameters to get final weights...")
    
    # Train using the entire available X_train
    if validation == "holdout":
        X_t_eval, y_t_eval = X_train, y_train
        X_v_eval, y_v_eval = X_val, y_val
    else:
        # For walk_forward global, X_val is None, X_train is the entire pool (0-90%).
        X_t_eval, y_t_eval = X_train.copy(), y_train.copy()
        X_v_eval, y_v_eval = None, None
        
    # Drop augmented correctly for final val if it exists
    if X_v_eval is not None and 'is_augmented' in X_v_eval.columns:
        y_v_eval = y_v_eval[X_v_eval['is_augmented'] == 0]
        X_v_eval = X_v_eval[X_v_eval['is_augmented'] == 0]
        
    # Weights for final train
    final_weights = generate_sample_weights(X_t_eval, y_t_eval, weight_strategy, upset_weight)

    # 4. Prepare data for evaluation
    drop_c = ['is_augmented', 'match_id', 'winner_name', 'loser_name']
    X_t_clean = X_t_eval.drop(columns=[c for c in drop_c if c in X_t_eval.columns])
    
    scaler = StandardScaler().fit(X_t_clean)
    X_t_scaled = scaler.transform(X_t_clean).astype(np.float32)
    y_t_np = y_t_eval.values.astype(np.float32)
    
    if X_v_eval is not None:
        X_v_clean = X_v_eval.drop(columns=[c for c in drop_c if c in X_v_eval.columns])
        X_v_scaled = scaler.transform(X_v_clean).astype(np.float32)
        y_v_np = y_v_eval.values.astype(np.float32)
    else:
        X_v_scaled = None
        y_v_np = None
    
    pc_cfg = PCNetworkConfig(
        learning_rate=best_params['learning_rate'],
        inference_lr=best_params['inference_lr'],
        inference_steps=best_params['inference_steps'],
        hidden_activation=best_params['hidden_activation'],
        output_activation="sigmoid",
        random_seed=42
    )
    
    depth = best_params['depth']
    width = best_params['width']
    hidden_sizes = [width] * depth
    layer_sizes = [X_t_scaled.shape[1], *hidden_sizes, 1]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=pc_cfg, device=device)
    
    # For global retrain without validation, train exactly to best_params['epochs']
    retrain_epochs = best_params['epochs']
    batch_size = best_params['batch_size']
    
    best_acc, best_ep, best_state, metrics_hist, train_time = train_pc_model_loop(
        model, X_t_scaled, y_t_np, X_v_scaled, y_v_np, 
        final_weights, max_epochs=retrain_epochs, batch_size=batch_size, verbose=True, patience=20
    )
    
    # Save artifacts
    model.load_state_dict(best_state)
    torch.save(best_state, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save training loop log text
    with open(log_path, 'w') as f:
        f.write("Predictive Coding Final Training Log\n")
        f.write(f"Hyperparameters: epochs={retrain_epochs}, batch_size={batch_size}, lr={best_params['learning_rate']:.4f}\n")
        f.write("="*80 + "\n")
        for m in metrics_hist:
            val_str = f" | Val Acc: {m['val_acc']:.4f}" if m['val_acc'] is not None else ""
            f.write(f"Epoch {m['epoch']:03d}/{retrain_epochs} | Train Energy: {m['train_energy']:.4f} | Train Loss (BCE): {m['train_loss']:.4f} | Train Acc: {m['train_acc']:.4f}{val_str}\n")
        f.write("="*80 + "\n")
        
        acc_label = "Val Acc" if (metrics_hist and metrics_hist[0]['val_acc'] is not None) else "Train Acc"
        f.write(f"Best {acc_label}: {best_acc:.4f} at epoch {best_ep}\n")
    
    config = {
        "model_type": "PredictiveCoding",
        "optimizer": optimizer,
        "best_params": {
            "learning_rate": best_params['learning_rate'],
            "inference_lr": best_params['inference_lr'],
            "inference_steps": best_params['inference_steps'],
            "hidden_activation": best_params['hidden_activation'],
            "random_seed": 42
        },
        "model_params": {
            "depth": depth,
            "width": width,
            "hidden_sizes": hidden_sizes,
            "training_params": {
                "batch_size": batch_size,
                "epochs": retrain_epochs,
                "weight_strategy": weight_strategy
            }
        },
        "evaluation": {
            "best_epoch": best_ep,
            "test_accuracy": best_acc,
            "training_time_seconds": train_time
        },
        "features_used": list(X_t_clean.columns)
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Artifacts saved to {output_dir}")
    
    # Generate Plots
    plot_optuna_history(study, reports_dir)
    plot_training_curve(metrics_hist, reports_dir)
    
    if X_v_scaled is not None:
        plot_feature_importance_permutation(model, X_v_scaled, y_v_np, X_v_clean.columns, reports_dir)
        from src.models.utils.metrics import binary_classification_metrics, evaluate_model_bias
        val_probs = model.predict_proba(X_v_scaled)
        val_preds = (val_probs >= 0.5).astype(int)
        final_metrics = binary_classification_metrics(y_v_np, val_probs)
        print("\nFINAL METRICS ON EVAL SET (Best Epoch):")
        for k, v in final_metrics.items():
            print(f" - {k:>16}: {v:.4f}")
            
        evaluate_model_bias(y_v_np, val_preds, X_v_clean, dataset_name="(VALIDATION SET)")
    
    return model, scaler

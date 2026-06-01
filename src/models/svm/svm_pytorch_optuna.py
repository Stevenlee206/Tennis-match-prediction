import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path
import copy 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, ConstantLR
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# PyTorch Model & Custom Loss
class PyTorchLinearSVM(nn.Module):
    def __init__(self, n_features):
        super(PyTorchLinearSVM, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze()
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def weighted_hinge_loss(predictions, targets, weights):
    """
    Hinge Loss: max(0, 1 - y_true * y_pred)
    Multiplied by custom sample weights.
    """
    raw_loss = torch.clamp(1 - predictions * targets, min=0)
    return torch.mean(raw_loss * weights)

# Weight Generation (Matched Logic)
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

# Plotting
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
        plt.savefig(save_path / "optuna_optimization_history_pytorch.png", dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    # Extract weights from the single linear layer
    importances = model.linear.weight.data.cpu().numpy()[0]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': importances,
        'Absolute_Importance': np.abs(importances)
    }).sort_values(by='Absolute_Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='Coefficient', y='Feature', palette="vlag")
    plt.title("PyTorch SVM Feature Importance (Linear Coefficients)")
    plt.xlabel("Coefficient Value (Directional Impact)")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path / "feature_importance.png", dpi=300)
    plt.close()

def plot_learning_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- Loss Plot ---
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.set_title('Training Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Hinge Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # --- Accuracy Plot ---
    ax2.plot(epochs, history['train_acc'], 'g-', label='Training Accuracy', linewidth=2)
    if len(history['val_acc']) > 0:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / "pytorch_learning_curves.png", dpi=300)
    plt.close()

# helper function
def get_optimizer_and_scheduler(model, lr, l2_lambda, opt_choice, sched_choice, epochs):
    """
    Init Optimizer and Scheduler.
    By default is SGD and Constant
    """
    if opt_choice == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    elif opt_choice == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_lambda)
    elif opt_choice == "sgd_nesterov":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=l2_lambda)
    else:
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_lambda)

    if sched_choice == "cosine":
        sched = CosineAnnealingLR(opt, T_max=epochs)
    elif sched_choice == "step":
        sched = StepLR(opt, step_size=10, gamma=0.5)
    elif sched_choice == "plateau":
        sched = ReduceLROnPlateau(opt, mode='max', patience=5, factor=0.5)
    else:
        sched = ConstantLR(opt, factor=1.0)
    return opt, sched

def apply_transforms(X_train, X_val=None, add_pca=False, add_kmeans=False, n_clusters=5):
    """Encapsulate logic Scaler, PCA and KMeans to reuse."""
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)
    X_val_proc = scaler.transform(X_val) if X_val is not None else None

    pca_model, kmeans_model = None, None

    if add_pca:
        pca_model = PCA(n_components=0.95, random_state=42)
        X_train_proc = pca_model.fit_transform(X_train_proc)
        if X_val_proc is not None:
            X_val_proc = pca_model.transform(X_val_proc)

    if add_kmeans:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        t_distances = kmeans_model.fit_transform(X_train_proc)
        X_train_proc = np.hstack((X_train_proc, t_distances))
        if X_val_proc is not None:
            v_distances = kmeans_model.transform(X_val_proc)
            X_val_proc = np.hstack((X_val_proc, v_distances))

    return X_train_proc, X_val_proc, scaler, pca_model, kmeans_model

def save_pipeline_artifacts(output_dir, reports_dir, model, scaler, pca, kmeans, study, history, config, feature_names):
    """Combine all file save and chart drawing commands into one place."""
    torch.save(model.state_dict(), output_dir / "svm_pytorch_model.pth")
    joblib.dump(scaler, output_dir / "svm_pytorch_scaler.joblib")
    if pca: joblib.dump(pca, output_dir / "svm_pytorch_pca.joblib")
    if kmeans: joblib.dump(kmeans, output_dir / "svm_pytorch_kmeans.joblib")

    with open(output_dir / "svm_pytorch_config.json", "w") as f:
        json.dump(config, f, indent=4)

    plot_optuna_history(study, reports_dir)
    plot_learning_curves(history, reports_dir)
    plot_feature_importance(model, feature_names, reports_dir)
# Core Training Function
def train_and_evaluate(X_train, y_train, X_val, y_val, params, epochs=50, batch_size=64, device="cpu", track_history=False):
    """Handles the actual PyTorch training loop for both holdout and CV folds."""
    
    y_train_svm = np.where(y_train == 0, -1, 1)
    weights_train = params["train_weights"]
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train_svm).to(device)
    w_train_t = torch.FloatTensor(weights_train).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t, w_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.FloatTensor(X_val).to(device)
    
    n_features = X_train.shape[1]
    model = PyTorchLinearSVM(n_features).to(device)
    
    l2_lambda = 1.0 / params["C"]
    lr = params["lr"]
    opt_choice = params["optimizer"]
    sched_choice = params["scheduler"]

    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        lr=lr,
        l2_lambda=l2_lambda,
        opt_choice=opt_choice,
        sched_choice=sched_choice,
        epochs=epochs
    )

    best_val_acc = 0.0
    best_epoch = epochs
    best_model_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = weighted_hinge_loss(preds, batch_y, batch_w)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)
            
        avg_train_loss = epoch_loss / len(X_train)
            
        # Validation & Metrics Step
        model.eval()
        with torch.no_grad():
            if track_history:
                train_preds_raw = model(X_train_t)
                train_preds_binary = (train_preds_raw > 0).cpu().numpy().astype(int)
                train_acc = accuracy_score(y_train, train_preds_binary)
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
            
            if has_val:
                val_preds = model(X_val_t)
                val_preds_binary = (val_preds > 0).cpu().numpy().astype(int)
                val_acc = accuracy_score(y_val, val_preds_binary)
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    best_model_weights = copy.deepcopy(model.state_dict())
                
                if track_history:
                    history['val_acc'].append(val_acc)
            else:
                val_acc = 0.0
        # Update Scheduler
        if sched_choice == "plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()

    if has_val:
        model.load_state_dict(best_model_weights)

    if track_history:
        return best_val_acc, best_epoch, model, history
    return best_val_acc, best_epoch, model

# Optimization Objective
def objective(trial, X_train_raw, y_train_raw, X_val_raw,
              y_val_raw, c_min, c_max, add_pca, add_kmeans,
              n_clusters, validation, weight_strategy, upset_weight,
              torch_opt, torch_sched, epochs, batch_size, device,
              n_splits=5, tscv_test_size=None):

    trial_seed = 42 + trial.number
    set_seed(trial_seed)
    
    # Save it so we can retrieve it later
    trial.set_user_attr("seed", trial_seed)
    params = {
        "C": trial.suggest_float("C", c_min, c_max, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "optimizer": torch_opt,
        "scheduler": torch_sched
    }

    if validation == "holdout":
        X_t_proc, X_v_proc, _, _, _ = apply_transforms(X_train_raw, X_val_raw, add_pca, add_kmeans, n_clusters)
        params["train_weights"] = generate_sample_weights(X_train_raw, y_train_raw, weight_strategy, upset_weight)

        best_val_acc, best_epoch, _ = train_and_evaluate(X_t_proc, y_train_raw, X_v_proc, y_val_raw, params, epochs,
                                                         batch_size, device)
        trial.set_user_attr("best_epoch", best_epoch)
        return best_val_acc

    elif validation == "walk_forward":
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)
        fold_accuracies, fold_best_epochs = [], []

        for train_index, val_index in tscv.split(X_train_raw):
            X_t_cv, X_v_cv = X_train_raw.iloc[train_index], X_train_raw.iloc[val_index]
            y_t_cv, y_v_cv = y_train_raw.iloc[train_index], y_train_raw.iloc[val_index]
            
            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t_cv)
            X_v_scaled = scaler.transform(X_v_cv)

            X_t_proc, X_v_proc, _, _, _ = apply_transforms(X_t_cv, X_v_cv, add_pca, add_kmeans, n_clusters)
            params["train_weights"] = generate_sample_weights(X_t_cv, y_t_cv, weight_strategy, upset_weight)

            fold_acc, best_epoch, _ = train_and_evaluate(X_t_proc, y_t_cv, X_v_proc, y_v_cv, params, epochs, batch_size,
                                                         device)
            fold_accuracies.append(fold_acc)
            fold_best_epochs.append(best_epoch)
            
        optimal_epoch = int(np.median(fold_best_epochs))
        trial.set_user_attr("best_epoch", optimal_epoch)
        
        return np.mean(fold_accuracies)

# Main Execution Pipeline
def run_pytorch_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, epochs=100, batch_size=64, c_min=1e-3, c_max=1e2, add_pca=False, add_kmeans=False, n_clusters=5, validation="holdout", weight_strategy="none", upset_weight=1.0, torch_opt="adam", torch_sched="cosine", n_splits=5, tscv_test_size=None):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing PyTorch Pipeline on: {device.upper()}")

    output_dir, reports_dir = Path(output_dir), Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "svm_pytorch_model.pth"
    scaler_path = output_dir / "svm_pytorch_scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        print(f"\n[!] Found existing artifacts in {output_dir.name}. Skipping training!")
        return None, joblib.load(scaler_path)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "svm_pytorch_optuna.db"
    
    print(f"\nStarting Optuna search ({n_trials} trials | Optimizer: {torch_opt.upper()} | Scheduler: {torch_sched.upper()})...")
    study = optuna.create_study(
        study_name="svm_pytorch_optimization",
        storage=f"sqlite:///{db_path.absolute()}",
        load_if_exists=True,
        direction="maximize"
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, c_min, c_max, add_pca, add_kmeans, n_clusters, validation, weight_strategy, upset_weight, torch_opt, torch_sched, epochs, batch_size, device, n_splits, tscv_test_size), 
        n_trials=n_trials
    )
    best_params = study.best_params
    best_params["optimizer"] = torch_opt
    best_params["scheduler"] = torch_sched
    
    # --- ADDED: Retrieve the exact epoch to stop at ---
    best_seed = study.best_trial.user_attrs["seed"]
    best_epoch_to_use = study.best_trial.user_attrs["best_epoch"] 
    
    set_seed(best_seed)
    
    print(f"\nBest Optuna parameters: {best_params}")
    print(f"Optimal Epochs (Early Stopping): {best_epoch_to_use}") # <--- LOG IT    
    # --- Final Training on Best Params ---
    print("Training final PyTorch model and generating learning curves...")
    # Chuẩn bị dữ liệu cho Final Training
    eval_y = y_val if validation == "holdout" and X_val is not None else None
    X_train_processed, X_val_processed, scaler, pca, kmeans = apply_transforms(X_train, X_val if eval_y is not None else None,
                                                                     add_pca, add_kmeans, n_clusters)
    best_params["train_weights"] = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
    # Call training with track_history=True
    _, _, final_model, history = train_and_evaluate(
        X_train_processed, y_train, X_val_processed, eval_y, 
        best_params, epochs=best_epoch_to_use, batch_size=batch_size, device=device, track_history=True
    )
    
    # Overfitting Check
    final_model.eval()
    with torch.no_grad():
        train_preds_raw = final_model(torch.FloatTensor(X_train_processed).to(device))
        train_preds_binary = (train_preds_raw > 0).cpu().numpy().astype(int)
        train_acc = accuracy_score(y_train, train_preds_binary)
        
    print("\n" + "-"*30)
    print(" OVERFITTING CHECK")
    print("-"*30)
    print(f"Training Accuracy:     {train_acc * 100:.2f}%")
    print(f"Optuna Val Accuracy:   {study.best_value * 100:.2f}%")
    if (train_acc - study.best_value) > 0.10:
        print("WARNING: High likelihood of overfitting.")
    print("-" * 30 + "\n")

    # Generate Plots
    print("Generating plots...")
    n_pcs = X_train_processed.shape[1] - (n_clusters if add_kmeans else 0)
    final_feature_names = [f"PC{i + 1}" for i in range(n_pcs)] if add_pca else list(X_train.columns)
    if add_kmeans: final_feature_names.extend([f"KMeans_Dist_C{i + 1}" for i in range(n_clusters)])

    config = {
        "model_type": "Linear-SVM (PyTorch)", "optimizer": torch_opt, "scheduler": torch_sched,
        "best_C": best_params["C"], "best_lr": best_params["lr"], "val_accuracy": study.best_value,
        "pca_applied": add_pca, "kmeans_applied": add_kmeans, "n_clusters": n_clusters if add_kmeans else 0,
        "weight_strategy": weight_strategy, "features_used": final_feature_names
    }

    # Save Artifacts
    save_pipeline_artifacts(output_dir, reports_dir, final_model, scaler, pca, kmeans, study, history, config,
                            final_feature_names)
    return final_model, scaler
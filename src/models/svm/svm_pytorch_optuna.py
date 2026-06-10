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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
# ==========================================
# PyTorch Model & Custom Loss
# ==========================================
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
# Plotting Utilities
# ==========================================
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
def apply_poly_kmeans(X_t, y_t, X_v=None, k_clusters=37, top_k=150):
    """Safely applies Poly Expansion -> Top-K Selection -> Scaler -> KMeans -> Dist Scaler"""
    # 1. Interaction-Only Polynomial Expansion
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_t_poly = poly.fit_transform(X_t)
    X_v_poly = poly.transform(X_v) if X_v is not None else None
    
    # 2. Feature Selection (Suppressing the expected divide-by-zero warnings from constant columns)
    selector = SelectKBest(score_func=f_classif, k=top_k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_t_sel = selector.fit_transform(X_t_poly, y_t)
    X_v_sel = selector.transform(X_v_poly) if X_v is not None else None
    
    # 3. Scale the selected poly features
    scaler_poly = StandardScaler()
    X_t_scaled = scaler_poly.fit_transform(X_t_sel)
    X_v_scaled = scaler_poly.transform(X_v_sel) if X_v is not None else None
    
    # 4. K-Means Clustering on the filtered interactions
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    t_dists = kmeans.fit_transform(X_t_scaled)
    
    # 5. Scale K-Means distances independently
    dist_scaler = StandardScaler()
    t_dists_scaled = dist_scaler.fit_transform(t_dists)
    
    # 6. Horizontally stack the base features with the distance features
    X_t_final = np.hstack((X_t_scaled, t_dists_scaled))
    
    if X_v is not None:
        v_dists = kmeans.transform(X_v_scaled)
        v_dists_scaled = dist_scaler.transform(v_dists)
        X_v_final = np.hstack((X_v_scaled, v_dists_scaled))
    else:
        X_v_final = None
        
    return X_t_final, X_v_final, poly, selector, scaler_poly, kmeans, dist_scaler
# ==========================================
# Core Training Function
# ==========================================
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
        y_val_svm = np.where(y_val == 0, -1, 1)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val_svm).to(device)  # ---> NEW: Create validation target tensor
    
    n_features = X_train.shape[1]
    model = PyTorchLinearSVM(n_features).to(device)
    
    l2_lambda = 1.0 / params["C"]
    lr = params["lr"]
    opt_choice = params["optimizer"]
    
    if opt_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    elif opt_choice == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_lambda)
    elif opt_choice == "sgd_nesterov":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=l2_lambda)
    else: 
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_lambda)

    sched_choice = params["scheduler"]
    if sched_choice == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_choice == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif sched_choice == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = ConstantLR(optimizer, factor=1.0) 

    best_val_acc = 0.0
    best_epoch = epochs  
    best_model_weights = copy.deepcopy(model.state_dict())
    
    # ---> UPDATED: Added 'val_loss' to the dictionary
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
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
                
                # ---> NEW: Calculate unweighted validation hinge loss
                val_weights = torch.ones_like(y_val_t).to(device) 
                v_loss = weighted_hinge_loss(val_preds, y_val_t, val_weights).item()
                
                val_preds_binary = (val_preds > 0).cpu().numpy().astype(int)
                val_acc = accuracy_score(y_val, val_preds_binary)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1 
                    best_model_weights = copy.deepcopy(model.state_dict())
                
                if track_history:
                    history['val_acc'].append(val_acc)
                    history['val_loss'].append(v_loss) # ---> NEW: Record the loss
            else:
                val_acc = 0.0
                
        if sched_choice == "plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()

    if has_val:
        model.load_state_dict(best_model_weights)

    if track_history:
        return best_val_acc, best_epoch, model, history
    return best_val_acc, best_epoch, model

# ==========================================
# Optimization Objective
# ==========================================
def objective(trial, X_train_raw, y_train_raw, X_val_raw, y_val_raw, c_min, c_max, add_pca, poly_features_kmeans, validation, weight_strategy, upset_weight, torch_opt, torch_sched, epochs, batch_size, device, n_splits=5, tscv_test_size=None):
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
        # ---> NEW ROUTING LOGIC <---
        if poly_features_kmeans:
            X_t_processed, X_v_processed, _, _, _, _, _ = apply_poly_kmeans(X_train_raw, y_train_raw, X_val_raw, k_clusters=37, top_k=150)
        else:
            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_train_raw)
            X_v_scaled = scaler.transform(X_val_raw)
            
            if add_pca:
                pca = PCA(n_components=0.95, random_state=42)
                X_t_processed = pca.fit_transform(X_t_scaled)
                X_v_processed = pca.transform(X_v_scaled)
            else:
                X_t_processed, X_v_processed = X_t_scaled, X_v_scaled

        params["train_weights"] = generate_sample_weights(X_train_raw, y_train_raw, weight_strategy, upset_weight)
        
        best_val_acc, best_epoch, _ = train_and_evaluate(X_t_processed, y_train_raw, X_v_processed, y_val_raw, params, epochs, batch_size, device, track_history=False)
        trial.set_user_attr("best_epoch", best_epoch)
        return best_val_acc

    elif validation == "walk_forward":
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=tscv_test_size)
        fold_accuracies = []
        fold_best_epochs = []
        
        for train_index, val_index in tscv.split(X_train_raw):
            X_t_cv = X_train_raw.iloc[train_index].copy()
            X_v_cv = X_train_raw.iloc[val_index].copy()
            y_t_cv = y_train_raw.iloc[train_index].copy()
            y_v_cv = y_train_raw.iloc[val_index].copy()
            
            # ---> FILTERING LOGIC FOR AUGMENTED DATA <---
            # ---> FILTERING LOGIC FOR AUGMENTED DATA <---
            if 'is_augmented' in X_v_cv.columns:
                val_mask = (X_v_cv['is_augmented'] == 0)
                X_v_cv = X_v_cv[val_mask]
                y_v_cv = y_v_cv[val_mask]
                
                X_t_cv = X_t_cv.drop(columns=['is_augmented'])
                X_v_cv = X_v_cv.drop(columns=['is_augmented'])
            
            # ---> FIX: ADDED MISSING ROUTING LOGIC HERE <---
            if poly_features_kmeans:
                X_t_processed, X_v_processed, _, _, _, _, _ = apply_poly_kmeans(X_t_cv, y_t_cv, X_v_cv, k_clusters=37, top_k=150)
            else:
                scaler = StandardScaler()
                X_t_scaled = scaler.fit_transform(X_t_cv)
                X_v_scaled = scaler.transform(X_v_cv)
                
                if add_pca:
                    pca = PCA(n_components=0.95, random_state=42)
                    X_t_processed = pca.fit_transform(X_t_scaled)
                    X_v_processed = pca.transform(X_v_scaled)
                else:
                    X_t_processed, X_v_processed = X_t_scaled, X_v_scaled

            params["train_weights"] = generate_sample_weights(X_t_cv, y_t_cv, weight_strategy, upset_weight)
            
            fold_acc, best_epoch, _ = train_and_evaluate(X_t_processed, y_t_cv, X_v_processed, y_v_cv, params, epochs, batch_size, device, track_history=False)
            
            fold_accuracies.append(fold_acc)
            fold_best_epochs.append(best_epoch) 
            
        optimal_epoch = int(np.median(fold_best_epochs))
        trial.set_user_attr("best_epoch", optimal_epoch)
        
        return np.mean(fold_accuracies)

# ==========================================
# Main Execution Pipeline
# ==========================================
def run_pytorch_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, n_trials=30, epochs=100, batch_size=64, c_min=1e-3, c_max=1e2, add_pca=False, poly_features_kmeans=False, validation="holdout", weight_strategy="none", upset_weight=1.0, torch_opt="adam", torch_sched="cosine", n_splits=5, tscv_test_size=None):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing PyTorch Pipeline on: {device.upper()}")

    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
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
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, c_min, c_max, add_pca, poly_features_kmeans, validation, weight_strategy, upset_weight, torch_opt, torch_sched, epochs, batch_size, device, n_splits, tscv_test_size), 
        n_trials=n_trials
    )
    best_params = study.best_params
    best_params["optimizer"] = torch_opt
    best_params["scheduler"] = torch_sched
    
    best_seed = study.best_trial.user_attrs["seed"]
    best_epoch_to_use = study.best_trial.user_attrs["best_epoch"] 
    
    set_seed(best_seed)
    
    print(f"\nBest Optuna parameters: {best_params}")
    print(f"Optimal Epochs (Early Stopping): {best_epoch_to_use}")    

    # --- Final Training on Best Params ---
    print("Training final PyTorch model and generating learning curves...")
    
    # Protect scaler from the augmented metadata flag
    # --- Final Training on Best Params ---
    if 'is_augmented' in X_train.columns:
        X_train_features = X_train.drop(columns=['is_augmented'])
    else:
        X_train_features = X_train

    # Prepare Validation Set for Final Training
    X_val_features = X_val if (validation == "holdout" and X_val is not None) else None

    # ---> NEW: FINAL DATA PREP & ARTIFACT EXPORT <---
    if poly_features_kmeans:
        print("Applying Poly-Interaction + K-Means Transformation...")
        X_train_processed, X_val_processed, poly, selector, scaler_poly, kmeans, dist_scaler = apply_poly_kmeans(
            X_train_features, y_train, X_val_features, k_clusters=37, top_k=150
        )
        
        # Save all the transformers
        joblib.dump(poly, output_dir / "svm_pytorch_poly.joblib")
        joblib.dump(selector, output_dir / "svm_pytorch_selector.joblib")
        joblib.dump(scaler_poly, output_dir / "svm_pytorch_scaler_poly.joblib")
        joblib.dump(kmeans, output_dir / "svm_pytorch_kmeans.joblib")
        joblib.dump(dist_scaler, output_dir / "svm_pytorch_dist_scaler.joblib")
        
        # Save a dummy base scaler so main.py file existence checks do not crash
        scaler = StandardScaler().fit(X_train_features)
        
        # Generate feature names accurately for the plot
        poly_feature_names = poly.get_feature_names_out(input_features=X_train_features.columns)
        selected_feature_names = poly_feature_names[selector.get_support(indices=True)]
        cluster_names = [f"Cluster_{i}_Dist" for i in range(37)]
        final_feature_names = list(selected_feature_names) + cluster_names

    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features) if X_val_features is not None else None
        
        if add_pca:
            pca = PCA(n_components=0.95, random_state=42)
            X_train_processed = pca.fit_transform(X_train_scaled)
            joblib.dump(pca, output_dir / "svm_pytorch_pca.joblib")
            X_val_processed = pca.transform(X_val_scaled) if X_val_scaled is not None else None
            final_feature_names = [f"PC{i+1}" for i in range(X_train_processed.shape[1])]
        else:
            X_train_processed = X_train_scaled
            X_val_processed = X_val_scaled
            final_feature_names = list(X_train_features.columns)

    if X_val_processed is None:
        eval_y = None
    else:
        eval_y = y_val

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
        print("⚠️ WARNING: High likelihood of overfitting.")
    print("-" * 30 + "\n")

    # Generate Plots
    # Generate Plots
    print("Generating plots...")
    plot_optuna_history(study, reports_dir)
    plot_learning_curves(history, reports_dir)
    
    # Just pass the already-defined final_feature_names directly!
    plot_feature_importance(final_model, final_feature_names, reports_dir)

    # Save Models
    torch.save(final_model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "Linear-SVM (PyTorch)",
        "optimizer": torch_opt,
        "scheduler": torch_sched,
        "best_C": best_params["C"],
        "best_lr": best_params["lr"],
        "val_accuracy": study.best_value,
        "poly_features_kmeans": poly_features_kmeans,
        "pca_applied": add_pca,
        "weight_strategy": weight_strategy,
        "features_used": final_feature_names
    }
    with open(output_dir / "svm_pytorch_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    return final_model, scaler
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# REUSABLE SAMPLE WEIGHT GENERATOR
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
# PYTORCH DATASET & MODEL
# ==========================================
class TennisDataset(Dataset):
    def __init__(self, X, y, weights=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        self.weights = torch.FloatTensor(weights).unsqueeze(1) if weights is not None else torch.ones_like(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate, activation_name):
        super().__init__()
        
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        activation_fn = activations.get(activation_name, nn.ReLU())
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        # Final output layer (Logits for BCEWithLogitsLoss)
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, epochs, device):
    criterion = nn.BCEWithLogitsLoss(reduction='none') 
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0  # <--- ADD THIS: Track the optimal epoch
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y, batch_w in train_loader:
            batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            
            # Apply custom upset weights to the loss
            loss = (criterion(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
            
        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y, _ in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_epoch = epoch  # <--- ADD THIS: Save the epoch number
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break 
                
    model.load_state_dict(best_model_state)
    # ---> FIX: Return both values
    return best_val_acc, best_epoch

def objective(trial, X_train, y_train, X_val, y_val, epochs, batch_size, add_pca=False, validation="holdout", weight_strategy="none", upset_weight=1.0):
    # ... (keep your existing setup logic) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_layers = [trial.suggest_int(f'n_units_l{i}', 16, 256, log=True) for i in range(n_layers)]
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    activation_name = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
    
    if validation == "holdout":
        scaler = StandardScaler()
        X_t_scaled = scaler.fit_transform(X_train)
        X_v_scaled = scaler.transform(X_val)
        
        if add_pca:
            pca = PCA(n_components=0.95, random_state=42)
            X_t_scaled = pca.fit_transform(X_t_scaled)
            X_v_scaled = pca.transform(X_v_scaled)
            
        weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
        
        train_dataset = TennisDataset(X_t_scaled, y_train.values, weights)
        val_dataset = TennisDataset(X_v_scaled, y_val.values)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = DynamicMLP(X_t_scaled.shape[1], hidden_layers, dropout_rate, activation_name).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_acc, best_epoch = train_model(model, train_loader, val_loader, optimizer, epochs, device)
        
        # ---> FIX: Save it to the Optuna trial
        trial.set_user_attr("best_epoch", best_epoch)
        
        return best_val_acc

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def run_pytorch_mlp_pipeline(X_train, y_train, X_val, y_val, output_dir, reports_dir, 
                             n_trials=30, epochs=100, batch_size=64, add_pca=False, 
                             validation="holdout", weight_strategy="none", upset_weight=1.0):
    
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = output_dir / "mlp_pytorch_model.pth"
    scaler_path = output_dir / "mlp_pytorch_scaler.joblib"

    print("Scaling features for PyTorch MLP...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if add_pca:
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        joblib.dump(pca, output_dir / "mlp_pytorch_pca.joblib")
    else:
        X_train_processed = X_train_scaled

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "mlp_pytorch_optuna.db"
    
    study = optuna.create_study(
        study_name=f"mlp_pytorch_{output_dir.name}",
        storage=f"sqlite:///{db_path.absolute()}",
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"\nStarting PyTorch MLP Optuna search ({n_trials} trials | Device: {device})...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, epochs, batch_size, add_pca, validation, weight_strategy, upset_weight),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    # ---> FIX: Extract the optimal epoch from the winning trial 
    # (+1 because epochs are 0-indexed)
    optimal_epochs = study.best_trial.user_attrs.get("best_epoch", epochs - 1) + 1
    print(f"Optimal epochs found via early stopping: {optimal_epochs}")
    
    # --- TRAIN FINAL MODEL ---
    hidden_layers = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    final_model = DynamicMLP(X_train_processed.shape[1], hidden_layers, best_params['dropout'], best_params['activation']).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
    train_dataset = TennisDataset(X_train_processed, y_train.values, weights)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    final_model.train()
    
    # ---> FIX: Use optimal_epochs instead of the max epochs
    for epoch in range(optimal_epochs):
        for batch_X, batch_y, batch_w in train_loader:
            batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
            optimizer.zero_grad()
            logits = final_model(batch_X)
            loss = (criterion(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
            
    torch.save(final_model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    # Plot Optuna History
    plt.figure(figsize=(10, 6))
    trials = study.trials_dataframe()
    if not trials.empty and "value" in trials.columns:
        sns.lineplot(data=trials, x="number", y="value", marker="o")
        plt.title("Optuna Optimization History (PyTorch MLP)")
        plt.savefig(reports_dir / "optuna_optimization_history.png", dpi=300)
    plt.close()

    # Save Config
    config = {
        "model_type": "PyTorch_MLP",
        "best_params": best_params,
        "hidden_layers": hidden_layers,
        "val_accuracy": study.best_value,
        "pca_applied": add_pca,
        "weight_strategy": weight_strategy
    }
    with open(output_dir / "mlp_pytorch_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    return final_model, scaler
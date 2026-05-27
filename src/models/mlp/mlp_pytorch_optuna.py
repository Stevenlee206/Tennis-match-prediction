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
import random
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
# PYTORCH DATASET & MODEL (FRIEND'S 1D CNN)
# ==========================================
class TimeSeriesTennisDataset(Dataset):
    def __init__(self, X, y, weights=None, window_size=5):
        """
        window_size: The number of consecutive matches the CNN will look at at once.
        Example with window_size=5: The model looks at matches t-4, t-3, t-2, t-1 to predict match t.
        """
        # Ensure X is a numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        self.weights = torch.FloatTensor(weights).unsqueeze(1) if weights is not None else torch.ones_like(self.y)
        self.window_size = window_size

    def __len__(self):
        # We lose the first few samples because they don't have enough history to form a full window
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        # Extract a temporal window of matches: from idx to idx + window_size
        window = self.X[idx : idx + self.window_size]

        # PyTorch Conv1d expects (Channels, Sequence_Length).
        # Here, Channels = Features (columns), Sequence_Length = Matches (rows in the window)
        # We transpose from (Matches, Features) to (Features, Matches)
        window = window.transpose(0, 1)

        # The target and weight belong strictly to the LAST match in this rolling window
        target_idx = idx + self.window_size - 1
        return window, self.y[target_idx], self.weights[target_idx]

class ReshapeInput(nn.Module): 
    def __init__(self):
        super(ReshapeInput, self).__init__()

    def forward(self, x):
        return x.unsqueeze(1)

class TimeSeriesTennisNet(nn.Module):
    def __init__(self, input_features, hidden_dim=64):
        super(TimeSeriesTennisNet, self).__init__()
        
        self.net = nn.Sequential(
            # --- CONV BLOCK 1 ---
            # in_channels MUST equal the number of features/columns in your dataset
            nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            # --- CONV BLOCK 2 ---
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.35),

            # --- CONV BLOCK 3 ---
            nn.Conv1d(in_channels=64, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            # Squash the temporal sequence dimension down to 1 before classifying
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # --- CLASSIFIER BLOCK ---
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            # Final Binary Output
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x arrives exactly as (Batch_Size, Features, Window_Size)
        return self.net(x)

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, epochs, device):
    criterion = nn.BCEWithLogitsLoss(reduction='none') 
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0 
    patience_counter = 0
    patience = 15
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y, batch_w in train_loader:
            batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            
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
            best_epoch = epoch  
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break 
                
    model.load_state_dict(best_model_state)
    return best_val_acc, best_epoch

def objective(trial, X_train_processed, y_train, X_val_processed, y_val, epochs, batch_size, validation, weight_strategy, upset_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # Set and save the seed
    seed = 42 + trial.number
    set_seed(seed)
    trial.set_user_attr("seed", seed)
    # --- HYPERPARAMETERS ---
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    if validation == "holdout":
        weights = generate_sample_weights(X_train_processed, y_train, weight_strategy, upset_weight)
        
        train_dataset = TimeSeriesTennisDataset(X_train_processed, y_train.values, weights)
        val_dataset = TimeSeriesTennisDataset(X_val_processed, y_val.values)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = TimeSeriesTennisNet(X_train_processed.shape[1], hidden_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_acc, best_epoch = train_model(model, train_loader, val_loader, optimizer, epochs, device)
        trial.set_user_attr("best_epoch", best_epoch)
        
        return best_val_acc

    elif validation == "walk_forward":
        tscv = TimeSeriesSplit(n_splits=5)
        fold_accuracies = []
        best_epochs = []
        
        for train_index, val_index in tscv.split(X_train_processed):
            X_t_cv = X_train_processed[train_index] if isinstance(X_train_processed, np.ndarray) else X_train_processed.iloc[train_index]
            y_t_cv = y_train.iloc[train_index]
            X_v_cv = X_train_processed[val_index] if isinstance(X_train_processed, np.ndarray) else X_train_processed.iloc[val_index]
            y_v_cv = y_train.iloc[val_index]
            
            weights = generate_sample_weights(X_t_cv, y_t_cv, weight_strategy, upset_weight)
            
            train_dataset = TimeSeriesTennisDataset(X_t_cv, y_t_cv.values, weights)
            val_dataset = TimeSeriesTennisDataset(X_v_cv, y_v_cv.values)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            model = TimeSeriesTennisNet(X_train_processed.shape[1], hidden_dim).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            val_acc, best_ep = train_model(model, train_loader, val_loader, optimizer, epochs, device)
            fold_accuracies.append(val_acc)
            best_epochs.append(best_ep)
            
        trial.set_user_attr("best_epoch", int(np.mean(best_epochs)))
        return np.mean(fold_accuracies)

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

    print("Scaling features for PyTorch Conv1D...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    if add_pca:
        pca = PCA(n_components=0.95, random_state=42)
        X_train_processed = pca.fit_transform(X_train_scaled)
        if X_val_scaled is not None:
            X_val_processed = pca.transform(X_val_scaled)
        else:
            X_val_processed = None
        joblib.dump(pca, output_dir / "mlp_pytorch_pca.joblib")
    else:
        X_train_processed = X_train_scaled
        X_val_processed = X_val_scaled

    optuna.logging.set_verbosity(optuna.logging.INFO)
    db_path = output_dir / "mlp_pytorch_optuna.db"
    
    study = optuna.create_study(
        study_name=f"mlp_pytorch_{output_dir.name}",
        storage=f"sqlite:///{db_path.absolute()}",
        load_if_exists=True,
        direction="maximize"
    )
    
    print(f"\nStarting PyTorch Conv1D Optuna search ({n_trials} trials | Device: {device})...")
    study.optimize(
        lambda trial: objective(trial, X_train_processed, y_train, X_val_processed, y_val, epochs, batch_size, validation, weight_strategy, upset_weight),
        n_trials=n_trials
    )
    
    best_params = study.best_params
    print(f"\nBest parameters found: {best_params}")
    
    optimal_epochs = study.best_trial.user_attrs.get("best_epoch", epochs - 1) + 1
    print(f"Optimal epochs found via early stopping: {optimal_epochs}")
    
    # Set seed for exact reproduction
    best_seed = study.best_trial.user_attrs.get("seed", 42)
    set_seed(best_seed)
    
    # --- TRAIN FINAL MODEL ---
    final_model = TimeSeriesTennisNet(X_train_processed.shape[1], best_params['hidden_dim']).to(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    weights = generate_sample_weights(X_train, y_train, weight_strategy, upset_weight)
    train_dataset = TimeSeriesTennisDataset(X_train_processed, y_train.values, weights)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    final_model.train()
    
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
        plt.title("Optuna Optimization History (PyTorch Conv1D)")
        plt.savefig(reports_dir / "optuna_optimization_history.png", dpi=300)
    plt.close()

    # Save Config
    config = {
        "model_type": "TennisNet_Conv1D (PyTorch)",
        "best_params": best_params,
        "hidden_dim": best_params['hidden_dim'],
        "val_accuracy": study.best_value,
        "pca_applied": add_pca,
        "weight_strategy": weight_strategy
    }
    with open(output_dir / "mlp_pytorch_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    return final_model, scaler
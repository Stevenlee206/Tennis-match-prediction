import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.predictive_coding.pc_network_torch import PredictiveCodingNetworkTorch
from src.models.predictive_coding.pc_network import PCNetworkConfig
from src.models.model import TennisNet

class NNWrapper:
    """
    Minimal wrapper for PyTorch NN to unify interface with PCN (predict_proba, train_on_batch).
    """
    def __init__(self, model: nn.Module, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().flatten()
        return probs
        
    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        logits = self.model(X_tensor)
        loss = self.criterion(logits, y_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

def get_nn_model(input_dim, mode, weights_path=None):
    """
    Initializes the NN. Loads weights if mode is static or finetune.
    """
    model = TennisNet(input_dim, hidden_dim=128, num_blocks=0, dropout=0.2)
    wrapper = NNWrapper(model=model, lr=0.001)
    
    if mode in ['static', 'finetune'] and weights_path and os.path.exists(weights_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wrapper.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded external NN weights from {weights_path}")
        
    return wrapper

def get_pcn_model(input_dim, mode, weights_path, config_path):
    """
    Initializes the PCN model directly. Loads weights if mode is static or finetune.
    During finetuning, learning rate is halved for stability.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_sizes = [input_dim, 64, 32, 1] # Default fallback
    cfg = PCNetworkConfig()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            if 'model_params' in config_dict and 'hidden_sizes' in config_dict['model_params']:
                hidden_sizes = config_dict['model_params']['hidden_sizes']
                layer_sizes = [input_dim] + hidden_sizes + [1]
                
            best_params = config_dict.get('best_params', {})
            cfg.learning_rate = best_params.get('learning_rate', 0.001)
            cfg.inference_lr = best_params.get('inference_lr', 0.05)
            cfg.inference_steps = best_params.get('inference_steps', 20)
            cfg.hidden_activation = best_params.get('hidden_activation', 'relu')
            cfg.output_activation = best_params.get('output_activation', 'sigmoid')
        except Exception as e:
            print(f"Failed to load PCN config: {e}. Using defaults.")
            
    # Reduce learning rate by half for fine-tuning
    if mode == 'finetune':
        cfg.learning_rate *= 0.5
        print(f"Fine-tune mode: Reduced PCN learning rate to {cfg.learning_rate}")
            
    model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=cfg, device=device)
    
    if mode in ['static', 'finetune'] and weights_path and os.path.exists(weights_path):
        try:
            state = np.load(weights_path, allow_pickle=True)
            state_dict = {k: state[k] for k in state.files}
            model.load_state_dict(state_dict)
            print(f"Loaded PCN weights from {weights_path}")
        except Exception as e:
            print(f"Failed to load PCN weights: {e}. Starting from scratch.")
            
    return model

def tune_hyperparameters_tscv(model_type, input_dim, X, y, n_splits=5, n_trials=10, epochs=50, patience=5, base_weights_path=None):
    """
    Uses Optuna and Time-Series Walk-Forward CV to find optimal hyperparameters (lr, weight_decay, etc.).
    Includes Early Stopping.
    """
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score

    print(f"\n--- Tuning {model_type.upper()} with Optuna (TSCV {n_splits}-fold, {n_trials} trials) ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # 1. Propose Hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        if model_type == "nn":
            wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
        else:
            # PCN
            inf_lr = trial.suggest_float("inference_lr", 0.01, 0.2, log=True)
            inf_steps = trial.suggest_int("inference_steps", 10, 50)
            
        # 2. Time-Series CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_accs = []
        fold_best_epochs = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Re-initialize model per fold
            if model_type == "nn":
                model = TennisNet(input_dim, hidden_dim=128, num_blocks=0, dropout=dropout)
                wrapper = NNWrapper(model, lr=lr)
                wrapper.optimizer.param_groups[0]['weight_decay'] = wd
                if base_weights_path and os.path.exists(base_weights_path):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    wrapper.load_state_dict(torch.load(base_weights_path, map_location=device, weights_only=True))
            elif model_type == "pcn":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cfg = PCNetworkConfig()
                cfg.learning_rate = lr
                cfg.inference_lr = inf_lr
                cfg.inference_steps = inf_steps
                cfg.output_activation = "identity"
                model = PredictiveCodingNetworkTorch(layer_sizes=[input_dim, 64, 32, 1], cfg=cfg, device=device)
                wrapper = model
                if base_weights_path and os.path.exists(base_weights_path):
                    state = np.load(base_weights_path, allow_pickle=True)
                    state_dict = {k: state[k] for k in state.files}
                    wrapper.load_state_dict(state_dict)

            # Early Stopping setup
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_fold_acc = 0
            best_epoch_for_fold = 1

            # 3. Train epochs
            for epoch in range(epochs):
                # Shuffle batch indices
                indices = np.random.permutation(len(X_train))
                X_shuf = X_train[indices]
                y_shuf = y_train[indices]
                
                batch_size = 64
                for i in range(0, len(X_train), batch_size):
                    X_b = X_shuf[i:i+batch_size]
                    y_b = y_shuf[i:i+batch_size]
                    wrapper.train_on_batch(X_b, y_b)
                
                # Validation
                probs = wrapper.predict_proba(X_val)
                y_pred = (probs >= 0.5).astype(int)
                val_acc = accuracy_score(y_val, y_pred)
                
                # We use -val_acc as loss proxy for early stopping
                val_loss_proxy = -val_acc
                
                if val_loss_proxy < best_val_loss:
                    best_val_loss = val_loss_proxy
                    best_fold_acc = val_acc
                    best_epoch_for_fold = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    break # Early stopping
            
            fold_accs.append(best_fold_acc)
            fold_best_epochs.append(best_epoch_for_fold)

        trial.set_user_attr("optimal_epochs", int(np.mean(fold_best_epochs)))
        return np.mean(fold_accs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_trial.params
    best_params["optimal_epochs"] = study.best_trial.user_attrs.get("optimal_epochs", epochs)
    
    print(f"Best trial: {study.best_trial.value:.4f}")
    print(f"Best params: {best_params}")
    return best_params

def train_model_full(model_type, input_dim, X: np.ndarray, y: np.ndarray, best_params, epochs=30, batch_size=64, base_weights_path=None):
    """
    Trains the model from scratch (or from base_weights) on the full X, y using best_params.
    Returns the trained model and history for plotting.
    """
    if model_type == "nn":
        model = TennisNet(input_dim, hidden_dim=128, num_blocks=0, dropout=best_params.get("dropout", 0.2))
        wrapper = NNWrapper(model, lr=best_params.get("lr", 0.001))
        wrapper.optimizer.param_groups[0]['weight_decay'] = best_params.get("wd", 0.0)
        
        if base_weights_path and os.path.exists(base_weights_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            wrapper.load_state_dict(torch.load(base_weights_path, map_location=device, weights_only=True))
            print(f"Loaded base NN weights for fine-tuning from {base_weights_path}")

    elif model_type == "pcn":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = PCNetworkConfig()
        cfg.learning_rate = best_params.get("lr", 0.001)
        cfg.inference_lr = best_params.get("inference_lr", 0.05)
        cfg.inference_steps = best_params.get("inference_steps", 20)
        cfg.output_activation = "identity"
        pcn_model = PredictiveCodingNetworkTorch(layer_sizes=[input_dim, 64, 32, 1], cfg=cfg, device=device)
        wrapper = pcn_model
        
        if base_weights_path and os.path.exists(base_weights_path):
            state = np.load(base_weights_path, allow_pickle=True)
            state_dict = {k: state[k] for k in state.files}
            wrapper.load_state_dict(state_dict)
            print(f"Loaded base PCN weights for fine-tuning from {base_weights_path}")

    history = {'loss': [], 'accuracy': []} # We'll just track training metrics since we train on full pool
    from sklearn.metrics import accuracy_score
    
    print(f"Training final {model_type.upper()} model for {epochs} epochs on {len(X)} samples...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        losses = []
        for i in range(0, len(X), batch_size):
            X_b = X_shuf[i:i+batch_size]
            y_b = y_shuf[i:i+batch_size]
            loss = wrapper.train_on_batch(X_b, y_b)
            losses.append(loss)
            
        epoch_loss = np.mean(losses)
        
        # Track training accuracy as well (a proxy for learning curve)
        probs = wrapper.predict_proba(X)
        y_pred = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f" Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {acc:.4f}")
            
    return wrapper, history

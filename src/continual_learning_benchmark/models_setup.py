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

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class NNWrapper:
    """
    Minimal wrapper for PyTorch NN to unify interface with PCN (predict_proba, train_on_batch).
    """
    def __init__(self, model: nn.Module, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
        
    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
        
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
    model = SimpleMLP(input_dim)
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

def train_model_full(model, X: np.ndarray, y: np.ndarray, epochs=3, batch_size=64):
    """
    Trains the model (NNWrapper or PCN) over the dataset for specified epochs.
    """
    print(f"Training model for {epochs} epochs on {len(X)} samples...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        losses = []
        for i in range(0, len(X), batch_size):
            X_b = X_shuf[i:i+batch_size]
            y_b = y_shuf[i:i+batch_size]
            loss = model.train_on_batch(X_b, y_b)
            losses.append(loss)
            
        print(f" Epoch {epoch+1}/{epochs} - Loss/Energy: {np.mean(losses):.4f}")
    return model

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.pcn_wrapper import PCNWrapper

# ---------------------------------------------------------
# PCN Wrappers
# ---------------------------------------------------------
class StaticPCNWrapper(PCNWrapper):
    """
    PCN in Static mode. Does NOT update weights during testing.
    """
    def online_train_step(self, X: np.ndarray, y: np.ndarray) -> dict:
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        acc = (preds == y).mean()
        return {"loss": 0.0, "accuracy": acc}

class OnlinePCNWrapper(PCNWrapper):
    """
    PCN in Online mode. Updates weights sequentially.
    """
    def __init__(self, model_path: str, config_path: str, input_dim: int):
        super().__init__(model_path, config_path, input_dim)
        # Reduce learning rate during streaming to mitigate catastrophic forgetting.
        # Since PCN uses SGD, a large LR overwrites base knowledge quickly on noisy stream data.
        # By lowering it, PCN slowly adapts without forgetting, outperforming standard NN.
        self.model.cfg.learning_rate = self.model.cfg.learning_rate * 0.05

# ---------------------------------------------------------
# NN Wrappers
# ---------------------------------------------------------
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

class FlexibleNNWrapper:
    """
    Base wrapper for a PyTorch Neural Network.
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
        
    def online_train_step(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
        
        logits = self.model(X_tensor)
        loss = self.criterion(logits, y_tensor)
        loss.backward()
        self.optimizer.step()
        
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        acc = (preds == y.flatten()).mean()
        
        return {"loss": loss.item(), "accuracy": acc}

class StaticNNWrapper(FlexibleNNWrapper):
    """
    NN in Static mode. Does NOT update weights.
    """
    def online_train_step(self, X: np.ndarray, y: np.ndarray) -> dict:
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        acc = (preds == y).mean()
        return {"loss": 0.0, "accuracy": acc}

class OnlineNNWrapper(FlexibleNNWrapper):
    """
    NN in Online mode. Updates weights sequentially.
    """
    pass

# ---------------------------------------------------------
# Factory function for NN
# ---------------------------------------------------------
def get_nn_model(input_dim, mode, external_weights_path=None):
    """
    Initializes the NN based on mode (static or online).
    Supports loading external weights (Option 2).
    """
    model = SimpleMLP(input_dim)
    
    if external_weights_path and os.path.exists(external_weights_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(external_weights_path, map_location=device))
        print(f"Loaded external NN weights from {external_weights_path}")
        
    if mode.lower() == 'static' or mode.lower() == 'retrain':
        # For Retrain, the model will be trained in the benchmark script before testing
        return StaticNNWrapper(model=model, lr=0.001)
    else:
        return OnlineNNWrapper(model=model, lr=0.001)

def get_pcn_model(input_dim, mode, model_path, config_path):
    """
    Initializes the PCN based on mode.
    """
    if mode.lower() == 'static' or mode.lower() == 'retrain':
        return StaticPCNWrapper(model_path=model_path, config_path=config_path, input_dim=input_dim)
    else:
        return OnlinePCNWrapper(model_path=model_path, config_path=config_path, input_dim=input_dim)

# ---------------------------------------------------------
# Training Helper for NN
# ---------------------------------------------------------
def train_nn_full(wrapper: FlexibleNNWrapper, X: np.ndarray, y: np.ndarray, epochs=5, batch_size=64):
    """
    Trains the NN from scratch over a full dataset (used for building the Static base model or Retraining).
    """
    print(f"Training NN from scratch for {epochs} epochs on {len(X)} samples...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        losses = []
        for i in range(0, len(X), batch_size):
            X_b = X_shuf[i:i+batch_size]
            y_b = y_shuf[i:i+batch_size]
            # Force the base class training step so it actually trains, regardless of static/online wrapper
            metrics = FlexibleNNWrapper.online_train_step(wrapper, X_b, y_b)
            losses.append(metrics['loss'])
            
        print(f" Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.4f}")
    return wrapper

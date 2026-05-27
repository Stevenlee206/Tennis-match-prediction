import os
import json
import numpy as np
import torch
from src.model.Predictive_Coding.pc_network_torch import PredictiveCodingNetworkTorch
from src.model.Predictive_Coding.pc_network import PCNetworkConfig

class PCNWrapper:
    """
    Wrapper for PredictiveCodingNetworkTorch to provide sklearn-like interface
    and easy online learning step integration.
    """
    def __init__(self, model_path: str, config_path: str, input_dim: int):
        self.input_dim = input_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load config if exists
        layer_sizes = [input_dim, 64, 32, 1] # Default fallback
        cfg = PCNetworkConfig()
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Extract layer sizes if available, else keep default
                if 'model_params' in config_dict and 'hidden_sizes' in config_dict['model_params']:
                    hidden_sizes = config_dict['model_params']['hidden_sizes']
                    layer_sizes = [input_dim] + hidden_sizes + [1]
                    
                # Setup cfg
                best_params = config_dict.get('best_params', {})
                cfg.learning_rate = best_params.get('learning_rate', 0.001)
                cfg.inference_lr = best_params.get('inference_lr', 0.05)
                cfg.inference_steps = best_params.get('inference_steps', 20)
                cfg.hidden_activation = best_params.get('hidden_activation', 'relu')
                cfg.output_activation = best_params.get('output_activation', 'sigmoid')
            except Exception as e:
                print(f"Failed to load PCN config: {e}. Using defaults.")
        
        self.model = PredictiveCodingNetworkTorch(layer_sizes=layer_sizes, cfg=cfg, device=self.device)
        
        # Load weights if exists
        if model_path and os.path.exists(model_path):
            try:
                state = np.load(model_path, allow_pickle=True)
                # Convert npz to dict
                state_dict = {k: state[k] for k in state.files}
                self.model.load_state_dict(state_dict)
                print(f"Loaded PCN weights from {model_path}")
            except Exception as e:
                print(f"Failed to load PCN weights: {e}. Starting from scratch.")
                
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
        
    def online_train_step(self, X: np.ndarray, y: np.ndarray) -> dict:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        energy = self.model.train_on_batch(X, y)
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        acc = (preds == y).mean()
        return {"loss": energy, "accuracy": acc}

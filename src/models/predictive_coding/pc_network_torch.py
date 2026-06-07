from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
import torch

from src.models.predictive_coding.pc_network import PCNetworkConfig
import src.models.preco.utils as preco_utils
import src.models.preco.optim as preco_optim
from src.models.preco.PCN import PCnet
from src.models.preco.structure import PCN_MBA

# Mapping string names to PRECO activation functions
act_map = {
    "tanh": preco_utils.tanh,
    "relu": preco_utils.relu,
    "sigmoid": preco_utils.sigmoid,
    "silu": preco_utils.silu,
    "identity": preco_utils.linear,
    "linear": preco_utils.linear,
    "leaky_relu": preco_utils.leaky_relu,
}

class PredictiveCodingNetworkTorch:
    """Predictive Coding network implemented using the PRECO library.
    
    Acts as a wrapper to maintain compatibility with the existing project structure.
    """
    is_torch = True

    def __init__(self, layer_sizes: Sequence[int], cfg: PCNetworkConfig, device: str | None = None):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output dimensions")

        self.layer_sizes = list(layer_sizes)
        self.cfg = cfg

        if device is not None:
            preco_utils.DEVICE = torch.device(device)
        self.device = preco_utils.DEVICE

        # Seed configuration
        preco_utils.seed(cfg.random_seed)

        # 1. Resolve activation functions (always use sigmoid output activation for classification)
        f_act = act_map.get(cfg.hidden_activation.lower(), preco_utils.tanh)
        fL_act = preco_utils.sigmoid

        # 2. Define structures (PCN_MBA is the standard multilayer prediction)
        structure = PCN_MBA(
            layers=self.layer_sizes,
            f=f_act,
            use_bias=True,
            upward=True,
            fL=fL_act
        )

        # 3. Create PRECO PCnet model
        self.pcnet = PCnet(
            lr_x=cfg.inference_lr,
            T_train=cfg.inference_steps,
            structure=structure,
            incremental=False,
            use_feedforward_init=True
        )

        # 4. Connect Adam Optimizer
        optimizer = preco_optim.Adam(
            self.pcnet.params,
            learning_rate=cfg.learning_rate,
            grad_clip=1.0,
            batch_scale=False,
            weight_decay=0.0
        )
        self.pcnet.set_optimizer(optimizer)

    def forward(self, x0: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run feedforward initialization.
        
        Returns states and dummy pre-activations to match the expected interface.
        """
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, device=self.device, dtype=torch.float32)
        else:
            x0 = x0.to(self.device, dtype=torch.float32)

        self.pcnet.reset_nodes()
        self.pcnet.clamp_input(x0)
        self.pcnet.forward(self.pcnet.error_layers)

        # We return the states list. Pre-activations are returned as dummy list of empty tensors.
        dummy_u = [torch.empty((x0.shape[0], 0), device=self.device) for _ in range(self.pcnet.L + 1)]
        return self.pcnet.x, dummy_u

    def get_state_dict(self) -> dict:
        """Returns standard state dict with W_l and b_l as PyTorch tensors."""
        state = {}
        for l in range(self.pcnet.L):
            # Transpose PRECO's (in_dim, out_dim) weight matrix to (out_dim, in_dim)
            W_pt = self.pcnet.w[l].detach().T.clone()
            b_pt = self.pcnet.b[l].detach().clone()
            state[f"W_{l}"] = W_pt
            state[f"b_{l}"] = b_pt
        return state

    def load_state_dict(self, state: dict) -> None:
        """Loads state dict with standard (out_dim, in_dim) weight layouts."""
        for l in range(self.pcnet.L):
            W = state[f"W_{l}"]
            b = state[f"b_{l}"]

            if not isinstance(W, torch.Tensor):
                W = torch.tensor(W, device=self.device, dtype=torch.float32)
            else:
                W = W.to(self.device, dtype=torch.float32)

            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b, device=self.device, dtype=torch.float32)
            else:
                b = b.to(self.device, dtype=torch.float32)

            # Transpose from standard (out_dim, in_dim) to PRECO's (in_dim, out_dim)
            self.pcnet.w[l] = W.T.clone()
            self.pcnet.b[l] = b.clone()

    def train_on_batch(self, x0: torch.Tensor, y: torch.Tensor, sample_weights: torch.Tensor | None = None) -> float:
        """Runs iterative inference followed by parameter updates."""
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, device=self.device, dtype=torch.float32)
        else:
            x0 = x0.to(self.device, dtype=torch.float32)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=torch.float32)
        else:
            y = y.to(self.device, dtype=torch.float32)

        if y.ndim == 1:
            y = y.view(-1, 1)

        # 1. Reset and clamp inputs
        self.pcnet.reset_nodes()
        self.pcnet.clamp_input(x0)
        self.pcnet.init_hidden(x0.shape[0])
        self.pcnet.clamp_target(y)

        # 2. Run standard iterative inference updates to optimize hidden states
        self.pcnet.train_updates()

        # 3. Apply sample weights if present
        if sample_weights is not None:
            if not isinstance(sample_weights, torch.Tensor):
                w = torch.tensor(sample_weights, device=self.device, dtype=torch.float32).view(-1, 1)
            else:
                w = sample_weights.to(self.device, dtype=torch.float32).view(-1, 1)
            self.pcnet.e[-1] = self.pcnet.e[-1] * w

        # 4. Local weight update
        self.pcnet.update_w()
        if not self.pcnet.incremental:
            self.pcnet.optimizer.step(self.pcnet.params, self.pcnet.grads, batch_size=x0.shape[0])

        # 5. Compute average energy across error layers (to match previous behaviour)
        energy = 0.0
        for l in self.pcnet.error_layers:
            e = self.pcnet.e[l]
            energy += 0.5 * float((e * e).mean().item())

        return energy

    def predict_proba_torch(self, x0: torch.Tensor) -> torch.Tensor:
        """Directly predict probabilities as PyTorch Tensor."""
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, device=self.device, dtype=torch.float32)
        else:
            x0 = x0.to(self.device, dtype=torch.float32)

        self.pcnet.reset_nodes()
        self.pcnet.clamp_input(x0)
        self.pcnet.forward(self.pcnet.error_layers)

        # Under the PCN_MBA structure, output states[-1] is already filtered by fL (sigmoid)
        out = self.pcnet.x[-1].view(-1)
        return out.clamp(1e-7, 1.0 - 1e-7)

    def predict_proba(self, x0: torch.Tensor) -> np.ndarray:
        """Predict win probabilities as numpy array."""
        return self.predict_proba_torch(x0).detach().cpu().numpy().astype(np.float32)

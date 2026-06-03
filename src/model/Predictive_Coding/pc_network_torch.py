from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore

from src.model.Predictive_Coding.pc_network import PCNetworkConfig


def _get_activation_torch(name: str):
    name = name.lower().strip()

    if torch is None:
        raise ImportError("PyTorch is required for PredictiveCodingNetworkTorch")

    if name == "relu":
        act = torch.relu

        def dact(u: torch.Tensor) -> torch.Tensor:
            return (u > 0).to(u.dtype)

        return act, dact

    if name == "tanh":
        act = torch.tanh

        def dact(u: torch.Tensor) -> torch.Tensor:
            y = torch.tanh(u)
            return 1.0 - y * y

        return act, dact

    if name == "sigmoid":
        act = torch.sigmoid

        def dact(u: torch.Tensor) -> torch.Tensor:
            s = torch.sigmoid(u)
            return s * (1.0 - s)

        return act, dact

    if name == "identity":
        def act(u: torch.Tensor) -> torch.Tensor:
            return u

        def dact(u: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(u)

        return act, dact

    raise KeyError(f"Unknown activation '{name}'.")


class _PCLayerTorch:
    def __init__(self, in_dim: int, out_dim: int, activation: str, rng: np.random.Generator, device: str):
        if torch is None:
            raise ImportError("PyTorch is required for _PCLayerTorch")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.activation_name = activation
        self.device = device

        self.act, self.dact = _get_activation_torch(activation)

        # Xavier/Glorot uniform init (match numpy implementation)
        limit = float(np.sqrt(6.0 / (self.in_dim + self.out_dim)))
        W_np = rng.uniform(-limit, limit, size=(self.out_dim, self.in_dim)).astype(np.float32)
        b_np = np.zeros((self.out_dim,), dtype=np.float32)

        self.W = torch.tensor(W_np, device=self.device, dtype=torch.float32)
        self.b = torch.tensor(b_np, device=self.device, dtype=torch.float32)

    def predict(self, x_in: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        # u = x @ W^T + b
        u = x_in @ self.W.t() + self.b
        return self.act(u), u

    def set_weights_from_numpy(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = torch.tensor(W.astype(np.float32), device=self.device, dtype=torch.float32)
        self.b = torch.tensor(b.astype(np.float32), device=self.device, dtype=torch.float32)

    def get_weights_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.W.detach().cpu().numpy().astype(np.float32), self.b.detach().cpu().numpy().astype(np.float32)


class PredictiveCodingNetworkTorch:
    """Predictive Coding network implemented in PyTorch (supports GPU).

    This keeps the same conceptual algorithm as the NumPy version:
    - Forward initializes states.
    - Iterative inference updates hidden states to reduce energy.
    - Local weight updates using prediction errors.

    Notes:
    - Uses manual tensor updates (no autograd required).
    - Saves/loads using NumPy-compatible state dict: {W_0, b_0, W_1, b_1, ...}
    """

    is_torch = True

    def __init__(self, layer_sizes: Sequence[int], cfg: PCNetworkConfig, device: str | None = None):
        if torch is None:
            raise ImportError("PyTorch is required for PredictiveCodingNetworkTorch")

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output")

        self.layer_sizes = list(layer_sizes)
        self.cfg = cfg

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Keep numpy RNG for identical init stats
        self.rng = np.random.default_rng(cfg.random_seed)

        self.layers: List[_PCLayerTorch] = []
        for i in range(len(layer_sizes) - 1):
            act = cfg.output_activation if i == len(layer_sizes) - 2 else cfg.hidden_activation
            self.layers.append(_PCLayerTorch(layer_sizes[i], layer_sizes[i + 1], act, rng=self.rng, device=self.device))

    def forward(self, x0: "torch.Tensor") -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
        x: List[torch.Tensor] = [x0.to(self.device, dtype=torch.float32)]
        u_list: List[torch.Tensor] = [torch.empty((x0.shape[0], 0), device=self.device, dtype=torch.float32)]

        for layer in self.layers:
            pred, u = layer.predict(x[-1])
            x.append(pred)
            u_list.append(u)

        return x, u_list

    def _predict_layer(self, layer_idx: int, x_in: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        return self.layers[layer_idx].predict(x_in)

    @torch.no_grad()
    def infer_hidden_states(self, states: List["torch.Tensor"], targets: "torch.Tensor", clamp_output: bool) -> None:
        if clamp_output:
            states[-1] = targets.to(self.device, dtype=torch.float32)

        L = len(self.layers)
        lr = float(self.cfg.inference_lr)

        for _ in range(int(self.cfg.inference_steps)):
            for l in range(1, L):
                # eps_l = x_l - f_{l-1}(W_{l-1} x_{l-1} + b)
                pred_l, _ = self._predict_layer(l - 1, states[l - 1])
                eps_l = states[l] - pred_l

                # For the output layer, no higher term
                if l == L:
                    states[l] = states[l] - lr * eps_l
                    continue

                pred_next, u_next = self._predict_layer(l, states[l])
                eps_next = states[l + 1] - pred_next
                delta_next = eps_next * self.layers[l].dact(u_next)
                back_term = delta_next @ self.layers[l].W

                dE_dx = eps_l - back_term
                states[l] = states[l] - lr * dE_dx

            if clamp_output:
                states[-1] = targets.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def _compute_errors(self, states: List["torch.Tensor"]):
        L = len(self.layers)
        eps: List[torch.Tensor] = [torch.empty((states[0].shape[0], 0), device=self.device, dtype=torch.float32)]
        u_list: List[torch.Tensor] = [torch.empty((states[0].shape[0], 0), device=self.device, dtype=torch.float32)]
        pred_list: List[torch.Tensor] = [torch.empty((states[0].shape[0], 0), device=self.device, dtype=torch.float32)]

        for l in range(L):
            pred, u = self._predict_layer(l, states[l])
            e = states[l + 1] - pred
            eps.append(e)
            u_list.append(u)
            pred_list.append(pred)

        return eps, u_list, pred_list

    def get_state_dict(self) -> dict:
        state = {}
        for i, layer in enumerate(self.layers):
            W_np, b_np = layer.get_weights_numpy()
            state[f"W_{i}"] = W_np
            state[f"b_{i}"] = b_np
        return state

    def load_state_dict(self, state: dict) -> None:
        for i, layer in enumerate(self.layers):
            W = state[f"W_{i}"]
            b = state[f"b_{i}"]

            if torch is not None and isinstance(W, torch.Tensor):
                W = W.detach().cpu().numpy()
            if torch is not None and isinstance(b, torch.Tensor):
                b = b.detach().cpu().numpy()

            layer.set_weights_from_numpy(W, b)

    @torch.no_grad()
    def train_on_batch(self, x0, y, sample_weights=None) -> float:
        if torch is None:
            raise ImportError("PyTorch is required for PredictiveCodingNetworkTorch")

        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, device=self.device, dtype=torch.float32)
        else:
            x0 = x0.to(self.device, dtype=torch.float32)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=torch.float32)
        else:
            y = y.to(self.device, dtype=torch.float32)

        states, _ = self.forward(x0)
        self.infer_hidden_states(states, targets=y, clamp_output=True)
        eps, u_list, _ = self._compute_errors(states)

        if sample_weights is not None:
            if not isinstance(sample_weights, torch.Tensor):
                w = torch.tensor(sample_weights, device=self.device, dtype=torch.float32).view(-1, 1)
            else:
                w = sample_weights.to(self.device, dtype=torch.float32).view(-1, 1)
            eps[-1] = eps[-1] * w

        lr = float(self.cfg.learning_rate)
        batch_size = float(x0.shape[0])

        energy = 0.0
        for l, layer in enumerate(self.layers):
            e = eps[l + 1]
            u = u_list[l + 1]
            delta = e * layer.dact(u)
            grad_W = (delta.t() @ states[l]) / batch_size
            grad_b = delta.mean(dim=0)

            layer.W = (layer.W + lr * grad_W).to(dtype=torch.float32)
            layer.b = (layer.b + lr * grad_b).to(dtype=torch.float32)

            energy += 0.5 * float((e * e).mean().item())

        return float(energy)

    @torch.no_grad()
    def predict_proba_torch(self, x0) -> "torch.Tensor":
        if torch is None:
            raise ImportError("PyTorch is required for PredictiveCodingNetworkTorch")

        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, device=self.device, dtype=torch.float32)
        else:
            x0 = x0.to(self.device, dtype=torch.float32)

        states, _ = self.forward(x0)
        out = states[-1].view(-1)
        return out.clamp(1e-7, 1.0 - 1e-7)

    @torch.no_grad()
    def predict_proba(self, x0) -> np.ndarray:
        return self.predict_proba_torch(x0).detach().cpu().numpy().astype(np.float32)

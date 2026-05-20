from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from pc_layer import PCLayer


@dataclass
class PCNetworkConfig:
	hidden_activation: str = "tanh"
	output_activation: str = "sigmoid"
	learning_rate: float = 1e-2
	inference_lr: float = 2e-1
	inference_steps: int = 20
	random_seed: int = 42


class PredictiveCodingNetwork:
	"""Predictive Coding network (supervised).

	Energy:
	E = 1/2 * Σ_l ||x_{l+1} - f_l(W_l x_l + b_l)||^2

	Training:
	- Clamp x_0 to inputs and x_L to targets.
	- Run iterative inference to update hidden states (x_1..x_{L-1}).
	- Update weights locally using prediction errors.
	"""

	def __init__(self, layer_sizes: Sequence[int], cfg: PCNetworkConfig):
		if len(layer_sizes) < 2:
			raise ValueError("layer_sizes must include input and output")
		self.layer_sizes = list(layer_sizes)
		self.cfg = cfg
		self.rng = np.random.default_rng(cfg.random_seed)

		self.layers: List[PCLayer] = []
		for i in range(len(layer_sizes) - 1):
			act = cfg.output_activation if i == len(layer_sizes) - 2 else cfg.hidden_activation
			self.layers.append(PCLayer(layer_sizes[i], layer_sizes[i + 1], act, rng=self.rng))

	def forward(self, x0: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		"""Feedforward initialization of states.

		Returns:
		- states x[0..L]
		- preactivations u[1..L] (u[0] unused)
		"""
		x: List[np.ndarray] = [x0.astype(np.float32)] # (batch, n_in)
		u_list: List[np.ndarray] = [np.empty((x0.shape[0], 0), dtype=np.float32)]
		for layer in self.layers:
			pred, u = layer.predict(x[-1])
			x.append(pred.astype(np.float32))
			u_list.append(u.astype(np.float32))
		return x, u_list

	def _predict_layer(self, layer_idx: int, x_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		return self.layers[layer_idx].predict(x_in)

	def infer_hidden_states(
		self,
		states: List[np.ndarray],
		targets: np.ndarray,
		clamp_output: bool,
	) -> None:
		"""Iterative inference to reduce energy by updating hidden states."""
		if clamp_output:
			states[-1] = targets.astype(np.float32)

		L = len(self.layers)
		lr = float(self.cfg.inference_lr)

		for _ in range(int(self.cfg.inference_steps)):
			# Update hidden layers only (1..L-1)
			for l in range(1, L):
				if l == L and clamp_output:
					continue
				# Compute eps_l = x_l - f_{l-1}(W_{l-1} x_{l-1} + b)
				pred_l, _ = self._predict_layer(l - 1, states[l - 1])
				eps_l = states[l] - pred_l

				# For the output layer, there is no higher term.
				if l == L:
					states[l] = (states[l] - lr * eps_l).astype(np.float32)
					continue

				# Compute back term from layer l+1
				pred_next, u_next = self._predict_layer(l, states[l])
				eps_next = states[l + 1] - pred_next
				delta_next = eps_next * self.layers[l].dact(u_next)
				back_term = delta_next @ self.layers[l].W  # (batch, dim_l)

				dE_dx = eps_l - back_term
				states[l] = (states[l] - lr * dE_dx).astype(np.float32)

			if clamp_output:
				states[-1] = targets.astype(np.float32)

	def _compute_errors(self, states: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
		"""Compute prediction errors for each layer.

		Returns (eps, u_list, pred_list) where indices correspond to x[1..L].
		- eps[k] corresponds to error at state x[k]
		- u_list[k] is preactivation producing prediction for x[k]
		- pred_list[k] is prediction for x[k]
		"""
		L = len(self.layers)
		eps: List[np.ndarray] = [np.empty((states[0].shape[0], 0), dtype=np.float32)]
		u_list: List[np.ndarray] = [np.empty((states[0].shape[0], 0), dtype=np.float32)]
		pred_list: List[np.ndarray] = [np.empty((states[0].shape[0], 0), dtype=np.float32)]
		for l in range(L):
			pred, u = self._predict_layer(l, states[l])
			e = (states[l + 1] - pred).astype(np.float32)
			eps.append(e)
			u_list.append(u.astype(np.float32))
			pred_list.append(pred.astype(np.float32))
		return eps, u_list, pred_list

	### UPDATE WEIGHTS ###
	def train_on_batch(self, x0: np.ndarray, y: np.ndarray) -> float:
		states, _ = self.forward(x0)
		self.infer_hidden_states(states, targets=y, clamp_output=True)

		eps, u_list, _ = self._compute_errors(states)

		# Local weight updates
		lr = float(self.cfg.learning_rate)
		batch_size = float(x0.shape[0])
		energy = 0.0
		for l, layer in enumerate(self.layers):
			e = eps[l + 1]
			u = u_list[l + 1]
			delta = e * layer.dact(u)
			grad_W = (delta.T @ states[l]) / batch_size
			grad_b = delta.mean(axis=0)
			layer.W = (layer.W + lr * grad_W).astype(np.float32)
			layer.b = (layer.b + lr * grad_b).astype(np.float32)
			energy += 0.5 * float(np.mean(e * e))

		return float(energy)

	def predict_proba(self, x0: np.ndarray) -> np.ndarray:
		states, _ = self.forward(x0)
		# Output activation is sigmoid by default; return probability of class 1.
		out = states[-1]
		return out.reshape(-1).astype(np.float32)

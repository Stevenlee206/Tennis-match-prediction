from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from src.models.pc.utils.activations import get_activation

Activation = Callable[[np.ndarray], np.ndarray]


@dataclass
class PCLayer:
	in_dim: int
	out_dim: int
	activation: str
	rng: np.random.Generator
	weight_scale: float = 1.0

	def __post_init__(self) -> None:
		act, dact = get_activation(self.activation)
		self.act: Activation = act
		self.dact: Activation = dact

		# Xavier/Glorot uniform init
		limit = self.weight_scale * np.sqrt(6.0 / (self.in_dim + self.out_dim)) # boundary
			# init W, b
		self.W = self.rng.uniform(-limit, limit, size=(self.out_dim, self.in_dim)).astype(np.float32)
		self.b = np.zeros((self.out_dim,), dtype=np.float32)

	def preact(self, x_in: np.ndarray) -> np.ndarray:
		return x_in @ self.W.T + self.b

	def predict(self, x_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		u = self.preact(x_in)
		return self.act(u), u

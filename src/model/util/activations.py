from __future__ import annotations

import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
	return np.maximum(0.0, x)

def drelu(x: np.ndarray) -> np.ndarray:
	return (x > 0).astype(x.dtype)


def tanh(x: np.ndarray) -> np.ndarray:
	return np.tanh(x)

def dtanh(x: np.ndarray) -> np.ndarray:
	y = np.tanh(x)
	return 1.0 - y * y


def sigmoid(x: np.ndarray) -> np.ndarray:
	# Numerically stable sigmoid: avoid overflow
	pos = x >= 0
	neg = ~pos

	out = np.empty_like(x, dtype=np.float32)
	out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

	exp_x = np.exp(x[neg])
	out[neg] = exp_x / (1.0 + exp_x)

	return out

def dsigmoid(x: np.ndarray) -> np.ndarray:
	s = sigmoid(x)
	return s * (1.0 - s)


def identity(x: np.ndarray) -> np.ndarray:
	return x

def didentity(x: np.ndarray) -> np.ndarray:
	return np.ones_like(x, dtype=x.dtype)

# activations and their derivatives
_ACTS = {
	"relu": (relu, drelu),
	"tanh": (tanh, dtanh),
	"sigmoid": (sigmoid, dsigmoid),
	"identity": (identity, didentity),
}


def get_activation(name: str):
	name = name.lower().strip()
	if name not in _ACTS:
		raise KeyError(f"Unknown activation '{name}'. Available: {list(_ACTS.keys())}")
	return _ACTS[name]

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.models.pc.utils.dataset import DatasetConfig, make_numpy_dataset
from src.models.pc.utils.metrics import binary_classification_metrics
from .pc_network import PCNetworkConfig, PredictiveCodingNetwork


@dataclass(frozen=True)
class TrainerConfig:
	epochs: int = 100
	batch_size: int = 256
	threshold: float = 0.5
	print_every: int = 1


class PCTrainer:
	def __init__(
		self,
		pc_cfg: PCNetworkConfig,
		trainer_cfg: TrainerConfig,
		ds_cfg: DatasetConfig,
		hidden_sizes: Tuple[int, ...] = (64, 32), # hidden layers
	) -> None:
		self.pc_cfg = pc_cfg
		self.trainer_cfg = trainer_cfg
		self.ds_cfg = ds_cfg
		self.hidden_sizes = hidden_sizes

		self.X_train, self.y_train, self.X_test, self.y_test, self.preprocessor = make_numpy_dataset(ds_cfg)
		layer_sizes = [self.X_train.shape[1], *list(hidden_sizes), 1]
		self.model = PredictiveCodingNetwork(layer_sizes=layer_sizes, cfg=pc_cfg)

	def _iterate_minibatches(self, X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
		idx = np.arange(X.shape[0])
		rng.shuffle(idx)
		for start in range(0, len(idx), batch_size):
			batch_idx = idx[start : start + batch_size]
			yb = y[batch_idx]
			xb = X[batch_idx]
			yb = yb.reshape(-1, 1).astype(np.float32)
			xb = xb.astype(np.float32)
			yield xb, yb

	def evaluate(self) -> Dict[str, Dict[str, float]]:
		train_prob = self.model.predict_proba(self.X_train)
		test_prob = self.model.predict_proba(self.X_test)

		return {
			"train": binary_classification_metrics(self.y_train, train_prob, threshold=self.trainer_cfg.threshold),
			"test": binary_classification_metrics(self.y_test, test_prob, threshold=self.trainer_cfg.threshold),
		}

	def fit(self) -> Dict[str, Dict[str, float]]:
		rng = np.random.default_rng(self.pc_cfg.random_seed)
		for epoch in range(1, int(self.trainer_cfg.epochs) + 1):
			energies = []
			for xb, yb in self._iterate_minibatches(self.X_train, self.y_train, self.trainer_cfg.batch_size, rng):
				energies.append(self.model.train_on_batch(xb, yb))

			if epoch % int(self.trainer_cfg.print_every) == 0:
				metrics = self.evaluate()
				print(
					f"Epoch {epoch:03d} | energy={float(np.mean(energies)):.4f} "
					f"| train_acc={metrics['train']['accuracy']:.4f} | train_logloss={metrics['train']['log_loss']:.4f}"
					f"| test_acc={metrics['test']['accuracy']:.4f} | test_auc={metrics['test']['roc_auc']:.4f} | test_logloss={metrics['test']['log_loss']:.4f}"
				)

		return self.evaluate()

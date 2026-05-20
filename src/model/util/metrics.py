from __future__ import annotations

from typing import Dict

import numpy as np


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
	from sklearn.metrics import (
		accuracy_score,
		average_precision_score,
		balanced_accuracy_score,
		brier_score_loss,
		f1_score,
		log_loss,
		precision_score,
		recall_score,
		roc_auc_score,
	)

	y_true = np.asarray(y_true).astype(int).reshape(-1)
	y_prob = np.asarray(y_prob).astype(float).reshape(-1)
	y_pred = (y_prob >= threshold).astype(int)

	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_true, y_prob)),
		"pr_auc": float(average_precision_score(y_true, y_prob)),
		"log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
		"brier": float(brier_score_loss(y_true, y_prob)),
	}


from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (e.g. `python decision_tree.py`) from within
# `src/model/` by ensuring the project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.preprocessing import Preprocessing


@dataclass(frozen=True)
class TrainConfig:
	label_column: str = "target"
	year_column: str = "year"
	train_start_year: int = 2014
	train_end_year: int = 2023
	test_year: int = 2024
	random_seed: int = 42

	# CART hyperparams (single decision tree)
	max_depth: int | None = 12
	min_examples: int = 10

	# Evaluation threshold for hard predictions
	threshold: float = 0.5


def _try_import_tfdf() -> Tuple[Optional[Any], Optional[Any]]:
	"""Try importing TensorFlow Decision Forests.

	Notes:
	- TF-DF provides Decision Tree models via a Keras API.
	- On Windows/Python 3.12 this may be unavailable due to wheel constraints.
	"""
	try:  # pragma: no cover
		import tensorflow as tf  # type: ignore
		import tensorflow_decision_forests as tfdf  # type: ignore
		return tf, tfdf
	except Exception:
		return None, None


def load_preprocessed_data() -> pd.DataFrame:
	"""Run the project's preprocessing pipeline and return a DataFrame."""
	df = Preprocessing().run()
	if df is None or len(df) == 0:
		raise ValueError("Preprocessing returned empty data")
	return df


def split_train_test_by_year(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if cfg.year_column not in df.columns:
		raise KeyError(
			f"Column '{cfg.year_column}' not found. "
			"Ensure preprocessing keeps 'year' (see src/preprocessing/feature_selection.py)."
		)

	train_mask = (df[cfg.year_column] >= cfg.train_start_year) & (df[cfg.year_column] <= cfg.train_end_year)
	test_mask = df[cfg.year_column] == cfg.test_year

	train_df = df.loc[train_mask].reset_index(drop=True)
	test_df = df.loc[test_mask].reset_index(drop=True)

	if len(train_df) == 0:
		raise ValueError("Train split is empty. Check year filtering / available data.")
	if len(test_df) == 0:
		raise ValueError("Test split is empty. Check year filtering / available data.")

	return train_df, test_df


def _prepare_for_tfdf(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
	"""Minimal cleaning so TF-DF can ingest the frame."""
	data = df.copy()
	# Don't leak the split key into the model unless you explicitly want it.
	if cfg.year_column in data.columns:
		data = data.drop(columns=[cfg.year_column])

	# Ensure label is int {0,1}
	if cfg.label_column not in data.columns:
		raise KeyError(f"Label column '{cfg.label_column}' not found")
	data[cfg.label_column] = data[cfg.label_column].astype(int)

	# TF-DF handles categoricals if dtype is string/category.
	for col in data.columns:
		if col == cfg.label_column:
			continue
		if pd.api.types.is_object_dtype(data[col]):
			data[col] = data[col].astype(str)

	return data


def _to_tf_dataset(tfdf: Any, df: pd.DataFrame, cfg: TrainConfig):
	return tfdf.keras.pd_dataframe_to_tf_dataset(
		df,
		label=cfg.label_column,
		task=tfdf.keras.Task.CLASSIFICATION,
	)


def train_decision_tree_keras(train_df: pd.DataFrame, cfg: TrainConfig):
	_, tfdf = _try_import_tfdf()
	if tfdf is None:
		raise ImportError(
			"tensorflow_decision_forests is not available in this environment. "
			"To use the Keras Decision Tree implementation, create an env with Python 3.10/3.11 "
			"and install tensorflow~=2.15 plus tensorflow_decision_forests."
		)

	train_data = _prepare_for_tfdf(train_df, cfg)
	train_ds = _to_tf_dataset(tfdf, train_data, cfg)

	model = tfdf.keras.CartModel(
		task=tfdf.keras.Task.CLASSIFICATION,
		max_depth=cfg.max_depth,
		min_examples=cfg.min_examples,
		random_seed=cfg.random_seed,
	)
	model.compile(metrics=["accuracy"])
	model.fit(train_ds, verbose=2)
	return model


def train_decision_tree_sklearn(train_df: pd.DataFrame, cfg: TrainConfig):
	"""Train a scikit-learn DecisionTreeClassifier with basic encoding.

	This is a fallback when TF-DF (Keras) isn't available on the current platform.
	"""
	from sklearn.compose import ColumnTransformer
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import OneHotEncoder
	from sklearn.tree import DecisionTreeClassifier

	data = train_df.copy()
	if cfg.year_column in data.columns:
		data = data.drop(columns=[cfg.year_column])

	y = data[cfg.label_column].astype(int).to_numpy()
	X = data.drop(columns=[cfg.label_column])

	cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
	num_cols = [c for c in X.columns if c not in cat_cols]

	# Fill missing values defensively.
	for c in cat_cols:
		X[c] = X[c].astype("string").fillna("Unknown")
	for c in num_cols:
		X[c] = pd.to_numeric(X[c], errors="coerce")
		if X[c].isna().any():
			X[c] = X[c].fillna(X[c].median())

	preprocess = ColumnTransformer(
		transformers=[
			("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
			("num", "passthrough", num_cols),
		],
		remainder="drop",
	)

	clf = DecisionTreeClassifier(
		max_depth=cfg.max_depth,
		min_samples_leaf=max(1, cfg.min_examples),
		random_state=cfg.random_seed,
	)

	model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
	model.fit(X, y)
	return model


def _binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
	"""Compute generalization-friendly metrics (threshold + threshold-free)."""
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

	y_true = np.asarray(y_true).astype(int)
	y_prob = np.asarray(y_prob).astype(float)
	y_pred = (y_prob >= threshold).astype(int)

	metrics: Dict[str, float] = {
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
	return metrics


def predict_proba(model: Any, df: pd.DataFrame, cfg: TrainConfig) -> np.ndarray:
	# scikit-learn pipeline/classifier
	if hasattr(model, "predict_proba"):
		data = df.copy()
		if cfg.year_column in data.columns:
			data = data.drop(columns=[cfg.year_column])
		X = data.drop(columns=[cfg.label_column])
		# Ensure consistent dtypes with training-time pipeline.
		for col in X.columns:
			if not pd.api.types.is_numeric_dtype(X[col]):
				X[col] = X[col].astype("string").fillna("Unknown")
		proba = model.predict_proba(X)[:, 1]
		return np.asarray(proba).reshape(-1)

	# TF-DF / Keras model
	_, tfdf = _try_import_tfdf()
	if tfdf is None:
		raise RuntimeError("Model does not support predict_proba and TF-DF is unavailable.")
	data = _prepare_for_tfdf(df, cfg)
	features = data.drop(columns=[cfg.label_column])
	predict_ds = tfdf.keras.pd_dataframe_to_tf_dataset(features, task=tfdf.keras.Task.CLASSIFICATION)
	proba = model.predict(predict_ds, verbose=0)
	return np.asarray(proba).reshape(-1)


def evaluate_model(model: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: TrainConfig) -> Dict[str, Dict[str, float]]:
	train_prob = predict_proba(model, train_df, cfg)
	test_prob = predict_proba(model, test_df, cfg)

	y_train = train_df[cfg.label_column].to_numpy()
	y_test = test_df[cfg.label_column].to_numpy()

	return {
		"train": _binary_classification_metrics(y_train, train_prob, cfg.threshold),
		"test": _binary_classification_metrics(y_test, test_prob, cfg.threshold),
	}


def train_and_evaluate(cfg: TrainConfig | None = None) -> Dict[str, Any]:
	"""End-to-end: preprocessing -> split -> train -> evaluate."""
	if cfg is None:
		cfg = TrainConfig()

	df = load_preprocessed_data()
	train_df, test_df = split_train_test_by_year(df, cfg)

	# Prefer Keras (TF-DF) if available; otherwise fallback to scikit-learn.
	try:
		model = train_decision_tree_keras(train_df, cfg)
		backend = "keras_tfdf"
	except ImportError:
		model = train_decision_tree_sklearn(train_df, cfg)
		backend = "sklearn"
	metrics = evaluate_model(model, train_df, test_df, cfg)

	return {
		"config": cfg,
		"n_train": int(len(train_df)),
		"n_test": int(len(test_df)),
		"metrics": metrics,
		"backend": backend,
		"model": model,
	}


def _print_metrics(title: str, d: Dict[str, float]) -> None:
	keys = [
		"accuracy",
		"balanced_accuracy",
		"precision",
		"recall",
		"f1",
		"roc_auc",
		"pr_auc",
		"log_loss",
		"brier",
	]
	print(f"\n{title}")
	for k in keys:
		if k in d:
			print(f"- {k:>16}: {d[k]:.5f}")


if __name__ == "__main__":
	result = train_and_evaluate()
	print(f"Backend: {result['backend']}")
	print(f"Train rows: {result['n_train']}, Test rows: {result['n_test']}")
	_print_metrics("Train metrics", result["metrics"]["train"])
	_print_metrics("Test metrics", result["metrics"]["test"])


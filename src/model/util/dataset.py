from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.preprocessing import Preprocessing


@dataclass(frozen=True)
class DatasetConfig:
	label_column: str = "target"
	year_column: str = "year"
	train_start_year: int = 2014
	train_end_year: int = 2023
	test_year: int = 2024


def load_preprocessed_df() -> pd.DataFrame:
	df = Preprocessing().run()
	if df is None or len(df) == 0:
		raise ValueError("Preprocessing returned empty data")	
	return df


def split_by_year(df: pd.DataFrame, cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
	if cfg.year_column not in df.columns:
		raise KeyError(
			f"Column '{cfg.year_column}' not found in preprocessed dataframe. "
			"Ensure preprocessing keeps 'year'."
		)
	train_mask = (df[cfg.year_column] >= cfg.train_start_year) & (df[cfg.year_column] <= cfg.train_end_year)
	test_mask = df[cfg.year_column] == cfg.test_year
	train_df = df.loc[train_mask].reset_index(drop=True)
	test_df = df.loc[test_mask].reset_index(drop=True)
	if len(train_df) == 0 or len(test_df) == 0:
		raise ValueError("Empty train/test split after filtering by years")
	return train_df, test_df


def build_feature_preprocessor(df: pd.DataFrame, cfg: DatasetConfig) -> ColumnTransformer:
	feature_cols = [c for c in df.columns if c not in {cfg.label_column, cfg.year_column}]
	X = df[feature_cols]

	cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
	num_cols = [c for c in feature_cols if c not in cat_cols]

	cat_pipe = Pipeline(
		steps=[
			("impute", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
		]
	)
	num_pipe = Pipeline(
		steps=[
			("impute", SimpleImputer(strategy="median")),
			("scale", StandardScaler()),
		]
	)

	ct = ColumnTransformer(
		transformers=[
			("cat", cat_pipe, cat_cols),
			("num", num_pipe, num_cols),
		],
		remainder="drop",
		verbose_feature_names_out=False,
	)
	return ct


def make_numpy_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
	"""Run preprocessing, split by years, and encode into NumPy arrays."""
	df = load_preprocessed_df()
	train_df, test_df = split_by_year(df, cfg)

	pre = build_feature_preprocessor(train_df, cfg)
	feature_cols = [c for c in train_df.columns if c not in {cfg.label_column, cfg.year_column}]

	X_train = pre.fit_transform(train_df[feature_cols])
	X_test = pre.transform(test_df[feature_cols])

	y_train = train_df[cfg.label_column].astype(np.float32).to_numpy().reshape(-1, 1)
	y_test = test_df[cfg.label_column].astype(np.float32).to_numpy().reshape(-1, 1)

	X_train = np.asarray(X_train, dtype=np.float32)
	X_test = np.asarray(X_test, dtype=np.float32)

	return X_train, y_train, X_test, y_test, pre

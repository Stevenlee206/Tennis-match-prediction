from __future__ import annotations

import os
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


MODE_LABELS = {
    "static": "Static",
    "finetune": "Finetune",
    "online": "Online",
    "ultimate_streaming": "Ultimate Streaming",
    "retrain": "Retrain Baseline",
}

MODE_ORDER = ["static", "finetune", "online", "ultimate_streaming", "retrain"]
HIGHER_IS_BETTER = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
}
LOWER_IS_BETTER = {"log_loss", "brier", "ece"}
EPS = 1e-15


def _warn_missing(columns: list[str], metric: str) -> None:
    warnings.warn(
        f"Skipping {metric}: missing optional column(s): {', '.join(columns)}",
        RuntimeWarning,
        stacklevel=2,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _safe_metric(fn: Callable, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return np.nan


def _clip_prob(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float).reshape(-1), EPS, 1.0 - EPS)


def normalize_prediction_records(records: pd.DataFrame) -> pd.DataFrame:
    df = records.copy()
    required = ["model_name", "mode", "y_true", "y_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction records are missing required columns: {missing}")

    df["mode"] = df["mode"].astype(str).str.lower()
    df["model_name"] = df["model_name"].astype(str).str.lower()
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = _clip_prob(df["y_prob"].values)
    if "y_pred" not in df.columns:
        df["y_pred"] = (df["y_prob"] >= 0.5).astype(int)
    else:
        df["y_pred"] = df["y_pred"].astype(int)
    if "match_id" not in df.columns:
        df["match_id"] = np.arange(len(df))
    if "prediction_before_update" not in df.columns:
        df["prediction_before_update"] = False
    return df


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = _clip_prob(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe_metric(average_precision_score, y_true, y_prob),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "class_1_prediction_rate": float(np.mean(y_pred == 1)),
    }


def compute_calibration_metrics(y_true, y_prob, y_pred=None, n_bins: int = 10) -> dict[str, object]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = _clip_prob(y_prob)
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correct = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidence = []
    bin_accuracy = []
    bin_count = []
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        count = int(mask.sum())
        bin_count.append(count)
        if count == 0:
            bin_confidence.append(np.nan)
            bin_accuracy.append(np.nan)
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(acc - conf)
        bin_accuracy.append(acc)
        bin_confidence.append(conf)
        ece += count / n * gap
        mce = max(mce, gap)

    wrong = y_pred != y_true
    overconfident_wrong_rate = np.nan
    if wrong.sum() > 0:
        overconfident_wrong_rate = float(np.mean(confidence[wrong] >= 0.8))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "mean_confidence": float(np.mean(confidence)),
        "std_confidence": float(np.std(confidence, ddof=0)),
        "overconfident_wrong_rate": overconfident_wrong_rate,
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_count": bin_count,
    }


def compute_prequential_metrics(df: pd.DataFrame) -> dict[str, float]:
    preq = df
    if "prediction_before_update" in df.columns:
        preq = df[df["prediction_before_update"].astype(bool)]
    if preq.empty:
        mode = df["mode"].iloc[0] if "mode" in df.columns and not df.empty else "unknown"
        model = df["model_name"].iloc[0] if "model_name" in df.columns and not df.empty else "unknown"
        warnings.warn(
            f"Skipping prequential metrics for {model}/{mode}: no prediction_before_update=True rows.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "prequential_accuracy": np.nan,
            "prequential_log_loss": np.nan,
            "prequential_brier": np.nan,
        }
    y_true = preq["y_true"].astype(int).values
    y_prob = _clip_prob(preq["y_prob"].values)
    y_pred = preq["y_pred"].astype(int).values
    return {
        "prequential_accuracy": float(accuracy_score(y_true, y_pred)),
        "prequential_log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "prequential_brier": float(brier_score_loss(y_true, y_prob)),
    }


def compute_time_weighted_metrics(df: pd.DataFrame, decay: float = 0.995) -> dict[str, float]:
    ordered = _sort_temporal(df)
    t = np.arange(len(ordered))
    weights = decay ** (len(ordered) - 1 - t)
    y_true = ordered["y_true"].astype(int).values
    y_prob = _clip_prob(ordered["y_prob"].values)
    y_pred = ordered["y_pred"].astype(int).values
    correct = (y_pred == y_true).astype(float)
    losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    briers = (y_prob - y_true) ** 2
    denom = weights.sum()
    return {
        "time_weighted_accuracy": float(np.sum(weights * correct) / denom),
        "time_weighted_log_loss": float(np.sum(weights * losses) / denom),
        "time_weighted_brier": float(np.sum(weights * briers) / denom),
    }


def _sort_temporal(df: pd.DataFrame) -> pd.DataFrame:
    date_col = next((c for c in ["date", "timestamp", "tourney_date"] if c in df.columns), None)
    if date_col is not None:
        ordered = df.copy()
        ordered[date_col] = pd.to_datetime(ordered[date_col], errors="coerce")
        return ordered.sort_values([date_col, "match_id"], kind="mergesort")
    if "online_step" in df.columns:
        return df.sort_values("online_step", kind="mergesort")
    if "update_step" in df.columns:
        return df.sort_values("update_step", kind="mergesort")
    return df.sort_values("match_id", kind="mergesort")


def compute_moving_window_metrics(
    df: pd.DataFrame, window_sizes: tuple[int, ...] = (100, 500)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows = []
    summary_rows = []
    for (model, mode), group in df.groupby(["model_name", "mode"], sort=False):
        ordered = _sort_temporal(group)
        y_true = ordered["y_true"].astype(int).values
        y_prob = _clip_prob(ordered["y_prob"].values)
        y_pred = ordered["y_pred"].astype(int).values
        losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        briers = (y_prob - y_true) ** 2
        correct = (y_pred == y_true).astype(float)
        x = _time_axis(ordered)
        for window in window_sizes:
            if len(ordered) < 2:
                continue
            actual_window = min(window, len(ordered))
            roll_acc = pd.Series(correct).rolling(actual_window, min_periods=1).mean()
            roll_ll = pd.Series(losses).rolling(actual_window, min_periods=1).mean()
            roll_brier = pd.Series(briers).rolling(actual_window, min_periods=1).mean()
            for i in range(len(ordered)):
                curve_rows.append(
                    {
                        "model": model,
                        "mode": MODE_LABELS.get(mode, mode),
                        "window_size": window,
                        "step": i,
                        "time": x.iloc[i] if isinstance(x, pd.Series) else x[i],
                        "moving_accuracy": float(roll_acc.iloc[i]),
                        "moving_log_loss": float(roll_ll.iloc[i]),
                        "moving_brier": float(roll_brier.iloc[i]),
                    }
                )
            summary_rows.append(
                {
                    "model": model,
                    "mode": MODE_LABELS.get(mode, mode),
                    "window_size": window,
                    "mean_moving_acc": float(roll_acc.mean()),
                    "best_moving_acc": float(roll_acc.max()),
                    "worst_moving_acc": float(roll_acc.min()),
                    "std_moving_acc": float(roll_acc.std(ddof=0)),
                    "mean_moving_logloss": float(roll_ll.mean()),
                    "worst_moving_logloss": float(roll_ll.max()),
                    "std_moving_logloss": float(roll_ll.std(ddof=0)),
                    "mean_moving_brier": float(roll_brier.mean()),
                    "worst_moving_brier": float(roll_brier.max()),
                    "std_moving_brier": float(roll_brier.std(ddof=0)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def _time_axis(df: pd.DataFrame):
    for col in ["date", "timestamp", "tourney_date"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce").reset_index(drop=True)
    return np.arange(len(df))


def compute_update_efficiency(
    df: pd.DataFrame, aggregate_metrics: pd.DataFrame, delta: float = 0.005
) -> pd.DataFrame:
    if not any(c in df.columns for c in ["update_time", "update_time_ms"]):
        _warn_missing(["update_time or update_time_ms"], "update efficiency")
        return pd.DataFrame()

    rows = []
    lookup = _metric_lookup(aggregate_metrics)
    for (model, mode), group in df.groupby(["model_name", "mode"], sort=False):
        if mode not in {"online", "ultimate_streaming", "finetune"}:
            continue
        time_col = "update_time_ms" if "update_time_ms" in group.columns else "update_time"
        values = pd.to_numeric(group[time_col], errors="coerce").dropna()
        mean_time = float(values.mean()) if len(values) else np.nan
        if time_col == "update_time_ms":
            mean_time_seconds = mean_time / 1000.0
        else:
            mean_time_seconds = mean_time
        static_acc = lookup.get((model, "static", "accuracy"), np.nan)
        mode_acc = lookup.get((model, mode, "accuracy"), np.nan)
        gain = mode_acc - static_acc if pd.notna(static_acc) and pd.notna(mode_acc) else np.nan
        ordered = _sort_temporal(group)
        correct = (ordered["y_pred"].astype(int) == ordered["y_true"].astype(int)).astype(float)
        initial_n = min(50, max(1, len(correct) // 10))
        initial_acc = float(correct.iloc[:initial_n].mean()) if len(correct) else np.nan
        samples_to_gain = np.nan
        for n in range(initial_n + 1, len(correct) + 1):
            if float(correct.iloc[:n].mean()) - initial_acc >= delta:
                samples_to_gain = n
                break
        updated_params = np.nan
        if "num_updated_params" in group.columns:
            updated_params = pd.to_numeric(group["num_updated_params"], errors="coerce").dropna().mean()
        rows.append(
            {
                "model": model,
                "mode": MODE_LABELS.get(mode, mode),
                "mean_update_time": mean_time_seconds,
                "samples_to_gain": samples_to_gain,
                "gain_per_update_time": _safe_divide(gain, mean_time_seconds),
                "performance_per_updated_param": _safe_divide(gain, updated_params),
            }
        )
    return pd.DataFrame(rows)


def compute_elo_rank_disagreement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "elo_pred" not in df.columns and "elo_diff" not in df.columns:
        _warn_missing(["elo_pred or elo_diff"], "Elo disagreement metrics")
    if "rank_pred" not in df.columns and "rank_diff" not in df.columns:
        _warn_missing(["rank_pred or rank_diff"], "Rank disagreement metrics")

    for (model, mode), group in df.groupby(["model_name", "mode"], sort=False):
        row = {"model": model, "mode": MODE_LABELS.get(mode, mode)}
        y_true = group["y_true"].astype(int)
        y_pred = group["y_pred"].astype(int)

        elo_pred = None
        if "elo_pred" in group.columns:
            elo_pred = group["elo_pred"].astype(int)
        elif "elo_diff" in group.columns:
            elo_pred = (pd.to_numeric(group["elo_diff"], errors="coerce") > 0).astype(int)
        if elo_pred is not None:
            agree = y_pred == elo_pred
            row["Elo Reliance"] = float(agree.mean())
            row["Elo Disagreement Rate"] = float((~agree).mean())
            row["Accuracy Agree Elo"] = float((y_pred[agree] == y_true[agree]).mean()) if agree.any() else np.nan
            row["Accuracy Disagree Elo"] = float((y_pred[~agree] == y_true[~agree]).mean()) if (~agree).any() else np.nan

        rank_pred = None
        if "rank_pred" in group.columns:
            rank_pred = group["rank_pred"].astype(int)
        elif "rank_diff" in group.columns:
            rank_pred = (pd.to_numeric(group["rank_diff"], errors="coerce") < 0).astype(int)
        if rank_pred is not None:
            agree = y_pred == rank_pred
            row["Rank Reliance"] = float(agree.mean())
            row["Rank Disagreement Rate"] = float((~agree).mean())
            row["Accuracy Agree Rank"] = float((y_pred[agree] == y_true[agree]).mean()) if agree.any() else np.nan
            row["Accuracy Disagree Rank"] = float((y_pred[~agree] == y_true[~agree]).mean()) if (~agree).any() else np.nan

        if "is_upset" in group.columns:
            upset = group["is_upset"].astype(bool)
        elif elo_pred is not None:
            upset = y_true != elo_pred
        else:
            upset = None
        if upset is not None:
            row["Upset Accuracy"] = float((y_pred[upset] == y_true[upset]).mean()) if upset.any() else np.nan
            row["Favorite Accuracy"] = float((y_pred[~upset] == y_true[~upset]).mean()) if (~upset).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    segment_cols = [c for c in ["test_year", "season", "surface", "tournament", "round", "is_upset"] if c in df.columns]
    if not segment_cols:
        _warn_missing(["test_year/season/surface/tournament/round/is_upset"], "segment-wise metrics")
        return pd.DataFrame()

    rows = []
    for segment_col in segment_cols:
        for (model, mode, value), group in df.groupby(["model_name", "mode", segment_col], dropna=False):
            metrics = compute_classification_metrics(group["y_true"], group["y_prob"])
            cal = compute_calibration_metrics(group["y_true"], group["y_prob"], group["y_pred"])
            rows.append(
                {
                    "segment": segment_col,
                    "segment_value": value,
                    "model": model,
                    "mode": MODE_LABELS.get(mode, mode),
                    "n": len(group),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                    "ece": cal["ece"],
                }
            )
    return pd.DataFrame(rows)


def compute_adaptation_speed(
    df: pd.DataFrame,
    window: int = 100,
    acc_alpha: float = 0.95,
    ll_beta: float = 1.05,
) -> pd.DataFrame:
    drift_cols = [c for c in ["test_year", "season", "surface", "tournament"] if c in df.columns]
    if not drift_cols:
        _warn_missing(["test_year/season/surface/tournament"], "adaptation speed")
        return pd.DataFrame()

    rows = []
    for (model, mode), group in df.groupby(["model_name", "mode"], sort=False):
        ordered = _sort_temporal(group).reset_index(drop=True)
        if len(ordered) < window * 2:
            continue
        y_true = ordered["y_true"].astype(int).values
        y_prob = _clip_prob(ordered["y_prob"].values)
        y_pred = ordered["y_pred"].astype(int).values
        correct = (y_pred == y_true).astype(float)
        losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        moving_acc = pd.Series(correct).rolling(window, min_periods=window).mean()
        moving_ll = pd.Series(losses).rolling(window, min_periods=window).mean()

        for drift_col in drift_cols:
            values = ordered[drift_col].reset_index(drop=True)
            drift_indices = values[values.ne(values.shift())].index.tolist()
            drift_indices = [idx for idx in drift_indices if idx >= window and idx < len(ordered) - 1]
            for tau in drift_indices:
                pre_acc = float(pd.Series(correct[:tau]).tail(window).mean())
                pre_ll = float(pd.Series(losses[:tau]).tail(window).mean())
                if pd.isna(pre_acc) or pd.isna(pre_ll):
                    continue
                acc_threshold = acc_alpha * pre_acc
                ll_threshold = ll_beta * pre_ll
                adaptation_time_acc = np.nan
                adaptation_time_ll = np.nan
                for k in range(1, len(ordered) - tau):
                    idx = tau + k
                    if pd.isna(adaptation_time_acc) and pd.notna(moving_acc.iloc[idx]) and moving_acc.iloc[idx] >= acc_threshold:
                        adaptation_time_acc = k
                    if pd.isna(adaptation_time_ll) and pd.notna(moving_ll.iloc[idx]) and moving_ll.iloc[idx] <= ll_threshold:
                        adaptation_time_ll = k
                    if pd.notna(adaptation_time_acc) and pd.notna(adaptation_time_ll):
                        break
                rows.append(
                    {
                        "model": model,
                        "mode": MODE_LABELS.get(mode, mode),
                        "drift_type": drift_col,
                        "drift_index": tau,
                        "drift_value": values.iloc[tau],
                        "pre_drift_accuracy": pre_acc,
                        "pre_drift_log_loss": pre_ll,
                        "adaptation_time_accuracy": adaptation_time_acc,
                        "adaptation_time_log_loss": adaptation_time_ll,
                        "window_size": window,
                    }
                )
    if not rows:
        warnings.warn(
            "Skipping adaptation speed: not enough post/pre-drift data for the configured window.",
            RuntimeWarning,
            stacklevel=2,
        )
    return pd.DataFrame(rows)


def compute_yearwise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    year_col = "test_year" if "test_year" in df.columns else "season" if "season" in df.columns else None
    if year_col is None:
        _warn_missing(["test_year or season"], "year-wise Static metrics")
        return pd.DataFrame()

    rows = []
    static_df = df[df["mode"] == "static"]
    for (model, year), group in static_df.groupby(["model_name", year_col]):
        metrics = compute_classification_metrics(group["y_true"], group["y_prob"])
        rows.append(
            {
                "model": model,
                "year": year,
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
            }
        )
    year_df = pd.DataFrame(rows)
    if year_df.empty:
        return year_df
    summary_rows = []
    for model, group in year_df.groupby("model"):
        summary_rows.append(
            {
                "model": model,
                "year": "summary",
                "mean_accuracy_across_years": group["accuracy"].mean(),
                "std_accuracy_across_years": group["accuracy"].std(ddof=0),
                "worst_year_accuracy": group["accuracy"].min(),
                "mean_log_loss_across_years": group["log_loss"].mean(),
                "std_log_loss_across_years": group["log_loss"].std(ddof=0),
                "worst_year_log_loss": group["log_loss"].max(),
                "mean_brier_across_years": group["brier"].mean(),
            }
        )
    return pd.concat([year_df, pd.DataFrame(summary_rows)], ignore_index=True)


def compute_aggregate_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    reliability_rows = []
    for (model, mode), group in df.groupby(["model_name", "mode"], sort=False):
        cls = compute_classification_metrics(group["y_true"], group["y_prob"])
        cal = compute_calibration_metrics(group["y_true"], group["y_prob"], group["y_pred"])
        preq = compute_prequential_metrics(group) if mode in {"online", "ultimate_streaming"} else {}
        tw = compute_time_weighted_metrics(group)
        row = {"model": model, "mode": mode, "n": len(group)}
        row.update(cls)
        row.update({k: v for k, v in cal.items() if not k.startswith("bin_")})
        row.update(preq)
        row.update(tw)
        metric_rows.append(row)
        for i, (conf, acc, count) in enumerate(zip(cal["bin_confidence"], cal["bin_accuracy"], cal["bin_count"])):
            reliability_rows.append(
                {
                    "model": model,
                    "mode": MODE_LABELS.get(mode, mode),
                    "bin": i,
                    "bin_confidence": conf,
                    "bin_accuracy": acc,
                    "bin_count": count,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(reliability_rows)


def _metric_lookup(metrics_df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    lookup = {}
    for _, row in metrics_df.iterrows():
        for metric in metrics_df.columns:
            if metric in {"model", "mode", "n"}:
                continue
            lookup[(row["model"], row["mode"], metric)] = row[metric]
    return lookup


def compute_adaptation_gains(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = _metric_lookup(metrics_df)
    models = sorted(metrics_df["model"].unique())
    rows = []
    for model in models:
        row = {"model": model}
        static_acc = lookup.get((model, "static", "accuracy"), np.nan)
        finetune_acc = lookup.get((model, "finetune", "accuracy"), np.nan)
        online_acc = lookup.get((model, "online", "accuracy"), np.nan)
        us_acc = lookup.get((model, "ultimate_streaming", "accuracy"), np.nan)
        retrain_acc = lookup.get((model, "retrain", "accuracy"), np.nan)
        row["Static Accuracy"] = static_acc
        row["Online Accuracy"] = online_acc
        row["Ultimate Streaming Accuracy"] = us_acc
        row["Online Adaptation Gain"] = online_acc - static_acc
        row["Ultimate Streaming Adaptation Gain"] = us_acc - static_acc
        row["Relative Online Gain"] = _safe_divide(online_acc - static_acc, static_acc) * 100
        row["Relative Ultimate Streaming Gain"] = _safe_divide(us_acc - static_acc, static_acc) * 100
        row["Ultimate Streaming Gain over Finetune"] = us_acc - finetune_acc
        row["Retrain Gap"] = us_acc - retrain_acc
        row["Online Gap to Retrain Accuracy"] = online_acc - retrain_acc
        row["US Gap to Retrain Accuracy"] = us_acc - retrain_acc

        static_ll = lookup.get((model, "static", "log_loss"), np.nan)
        finetune_ll = lookup.get((model, "finetune", "log_loss"), np.nan)
        online_ll = lookup.get((model, "online", "log_loss"), np.nan)
        us_ll = lookup.get((model, "ultimate_streaming", "log_loss"), np.nan)
        retrain_ll = lookup.get((model, "retrain", "log_loss"), np.nan)
        row["Online Adaptation Gain LogLoss"] = static_ll - online_ll
        row["Ultimate Streaming Adaptation Gain LogLoss"] = static_ll - us_ll
        row["Relative Online Gain LogLoss"] = _safe_divide(static_ll - online_ll, static_ll) * 100
        row["Relative Ultimate Streaming Gain LogLoss"] = _safe_divide(static_ll - us_ll, static_ll) * 100
        row["Ultimate Streaming Gain over Finetune LogLoss"] = finetune_ll - us_ll
        row["Online Gap to Retrain LogLoss"] = retrain_ll - online_ll
        row["US Gap to Retrain LogLoss"] = retrain_ll - us_ll

        static_brier = lookup.get((model, "static", "brier"), np.nan)
        us_brier = lookup.get((model, "ultimate_streaming", "brier"), np.nan)
        retrain_brier = lookup.get((model, "retrain", "brier"), np.nan)
        row["Ultimate Streaming Adaptation Gain Brier"] = static_brier - us_brier
        row["US Gap to Retrain Brier"] = retrain_brier - us_brier

        retrain_ft_acc = lookup.get((model, "retrain", "accuracy"), np.nan)
        row["Fine-tuning Gain Accuracy"] = finetune_acc - static_acc
        row["Relative Fine-tuning Gain Accuracy"] = _safe_divide(finetune_acc - static_acc, static_acc) * 100
        row["Fine-tuning Efficiency Accuracy"] = _safe_divide(finetune_acc, retrain_ft_acc)
        row["Fine-tuning Gap to Retrain Accuracy"] = finetune_acc - retrain_ft_acc
        row["Fine-tuning Gain LogLoss"] = static_ll - finetune_ll
        row["Relative Fine-tuning Gain LogLoss"] = _safe_divide(static_ll - finetune_ll, static_ll) * 100
        row["Fine-tuning Efficiency LogLoss"] = _safe_divide(retrain_ll, finetune_ll)
        row["Fine-tuning Gap to Retrain LogLoss"] = retrain_ll - finetune_ll
        rows.append(row)

    gains_df = pd.DataFrame(rows)
    if "pcn" in models:
        nn_name = "nn" if "nn" in models else next((m for m in models if m != "pcn"), None)
        if nn_name is not None:
            pcn_gain = gains_df.loc[gains_df["model"] == "pcn", "Ultimate Streaming Adaptation Gain"].iloc[0]
            nn_gain = gains_df.loc[gains_df["model"] == nn_name, "Ultimate Streaming Adaptation Gain"].iloc[0]
            pcn_ll_gain = gains_df.loc[gains_df["model"] == "pcn", "Ultimate Streaming Adaptation Gain LogLoss"].iloc[0]
            nn_ll_gain = gains_df.loc[gains_df["model"] == nn_name, "Ultimate Streaming Adaptation Gain LogLoss"].iloc[0]
            gains_df.loc[gains_df["model"] == "pcn", "Adaptation Gain Advantage over NN"] = pcn_gain - nn_gain
            gains_df.loc[gains_df["model"] == "pcn", "Adaptation Gain Advantage over NN LogLoss"] = pcn_ll_gain - nn_ll_gain

    comparison_rows = []
    if "pcn" in models:
        for other in [m for m in models if m != "pcn"]:
            row = {"comparison": f"pcn_vs_{other}", "mode": "Ultimate Streaming"}
            for metric in HIGHER_IS_BETTER:
                row[f"PCN_Streaming_Advantage_{metric}"] = (
                    lookup.get(("pcn", "ultimate_streaming", metric), np.nan)
                    - lookup.get((other, "ultimate_streaming", metric), np.nan)
                )
            row["PCN_Streaming_Advantage_log_loss"] = (
                lookup.get((other, "ultimate_streaming", "log_loss"), np.nan)
                - lookup.get(("pcn", "ultimate_streaming", "log_loss"), np.nan)
            )
            row["PCN_Streaming_Advantage_brier"] = (
                lookup.get((other, "ultimate_streaming", "brier"), np.nan)
                - lookup.get(("pcn", "ultimate_streaming", "brier"), np.nan)
            )
            row["PCN_Adaptation_Gain_Advantage_accuracy"] = (
                lookup.get(("pcn", "ultimate_streaming", "accuracy"), np.nan)
                - lookup.get(("pcn", "static", "accuracy"), np.nan)
            ) - (
                lookup.get((other, "ultimate_streaming", "accuracy"), np.nan)
                - lookup.get((other, "static", "accuracy"), np.nan)
            )
            row["PCN_Adaptation_Gain_Advantage_log_loss"] = (
                lookup.get(("pcn", "static", "log_loss"), np.nan)
                - lookup.get(("pcn", "ultimate_streaming", "log_loss"), np.nan)
            ) - (
                lookup.get((other, "static", "log_loss"), np.nan)
                - lookup.get((other, "ultimate_streaming", "log_loss"), np.nan)
            )
            comparison_rows.append(row)
    return gains_df, pd.DataFrame(comparison_rows)


def bootstrap_confidence_interval(
    y_true,
    y_prob,
    metric: str,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = _clip_prob(y_prob)
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        values.append(_single_metric(y_true[idx], y_prob[idx], metric))
    alpha = (1 - ci) / 2
    return (
        _single_metric(y_true, y_prob, metric),
        float(np.nanquantile(values, alpha)),
        float(np.nanquantile(values, 1 - alpha)),
    )


def _single_metric(y_true: np.ndarray, y_prob: np.ndarray, metric: str) -> float:
    y_pred = (y_prob >= 0.5).astype(int)
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "roc_auc":
        return _safe_metric(roc_auc_score, y_true, y_prob)
    if metric == "pr_auc":
        return _safe_metric(average_precision_score, y_true, y_prob)
    if metric == "log_loss":
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    if metric == "brier":
        return float(brier_score_loss(y_true, y_prob))
    if metric == "ece":
        return float(compute_calibration_metrics(y_true, y_prob)["ece"])
    raise ValueError(f"Unsupported bootstrap metric: {metric}")


def compute_bootstrap_ci_table(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier",
        "ece",
    ),
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for (model, mode), group in df.groupby(["model_name", "mode"], sort=False):
        for metric in metrics:
            mean, lower, upper = bootstrap_confidence_interval(
                group["y_true"].values,
                group["y_prob"].values,
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
            rows.append(
                {
                    "model": model,
                    "mode": MODE_LABELS.get(mode, mode),
                    "metric": metric,
                    "metric_mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def compute_paired_bootstrap_comparison_ci(
    df: pd.DataFrame,
    baseline_models: tuple[str, ...] = ("nn", "resnet"),
    mode: str = "ultimate_streaming",
    metrics: tuple[str, ...] = ("accuracy", "log_loss", "brier", "f1", "roc_auc", "pr_auc"),
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    if "pcn" not in set(df["model_name"]):
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(seed)
    for baseline in baseline_models:
        if baseline not in set(df["model_name"]):
            continue
        pcn = df[(df["model_name"] == "pcn") & (df["mode"] == mode)]
        other = df[(df["model_name"] == baseline) & (df["mode"] == mode)]
        matched = pcn.merge(
            other,
            on="match_id",
            suffixes=("_pcn", f"_{baseline}"),
            how="inner",
        )
        if matched.empty:
            continue
        y_true = matched["y_true_pcn"].astype(int).values
        p_pcn = _clip_prob(matched["y_prob_pcn"].values)
        p_other = _clip_prob(matched[f"y_prob_{baseline}"].values)
        n = len(matched)
        for metric in metrics:
            value = _paired_metric_diff(y_true, p_pcn, p_other, metric)
            values = []
            for _ in range(n_resamples):
                idx = rng.integers(0, n, n)
                values.append(_paired_metric_diff(y_true[idx], p_pcn[idx], p_other[idx], metric))
            rows.append(
                {
                    "comparison": f"pcn_vs_{baseline}",
                    "mode": MODE_LABELS.get(mode, mode),
                    "metric": metric,
                    "positive_means_pcn_better": True,
                    "metric_mean": value,
                    "ci_lower": float(np.nanquantile(values, 0.025)),
                    "ci_upper": float(np.nanquantile(values, 0.975)),
                    "n_matched": n,
                }
            )
    return pd.DataFrame(rows)


def _paired_metric_diff(y_true: np.ndarray, p_pcn: np.ndarray, p_other: np.ndarray, metric: str) -> float:
    if metric in LOWER_IS_BETTER:
        return _single_metric(y_true, p_other, metric) - _single_metric(y_true, p_pcn, metric)
    return _single_metric(y_true, p_pcn, metric) - _single_metric(y_true, p_other, metric)


def build_executive_summary(
    aggregate: pd.DataFrame,
    gains: pd.DataFrame,
    streaming_comparison: pd.DataFrame,
    elo_rank: pd.DataFrame,
    efficiency: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    lookup = _metric_lookup(aggregate)
    models = sorted(aggregate["model"].unique()) if not aggregate.empty else []
    baseline = "nn" if "nn" in models else next((m for m in models if m != "pcn"), None)

    def add(question, evidence, conclusion, action):
        rows.append(
            {
                "question": question,
                "evidence": evidence,
                "conclusion": conclusion,
                "recommended_claim": action,
            }
        )

    if "pcn" in models and baseline is not None:
        pcn_us_acc = lookup.get(("pcn", "ultimate_streaming", "accuracy"), np.nan)
        base_us_acc = lookup.get((baseline, "ultimate_streaming", "accuracy"), np.nan)
        pcn_static_acc = lookup.get(("pcn", "static", "accuracy"), np.nan)
        base_static_acc = lookup.get((baseline, "static", "accuracy"), np.nan)
        pcn_gain = pcn_us_acc - pcn_static_acc
        base_gain = base_us_acc - base_static_acc
        gain_adv = pcn_gain - base_gain
        add(
            "Does PCN improve hard-label winner prediction in Ultimate Streaming?",
            (
                f"PCN US Acc={_fmt_pct(pcn_us_acc)}, {baseline.upper()} US Acc={_fmt_pct(base_us_acc)}; "
                f"PCN Static->US Gain={_fmt_pct_delta(pcn_gain)}."
            ),
            _sign_conclusion(
                pcn_us_acc - base_us_acc,
                "PCN has higher Ultimate Streaming accuracy.",
                f"{baseline.upper()} has higher Ultimate Streaming accuracy.",
                "Ultimate Streaming accuracy is effectively tied.",
            ),
            (
                "Claim PCN is better for streaming hard-label prediction."
                if pcn_us_acc > base_us_acc
                else "Do not claim PCN beats the baseline on final hard-label accuracy."
            ),
        )
        add(
            "Does PCN benefit more from sequential updates than the baseline?",
            (
                f"PCN Static->US Gain={_fmt_pct_delta(pcn_gain)}; "
                f"{baseline.upper()} Static->US Gain={_fmt_pct_delta(base_gain)}; "
                f"Gain Advantage={_fmt_pct_delta(gain_adv)}."
            ),
            _sign_conclusion(
                gain_adv,
                "PCN benefits more from streaming adaptation.",
                f"{baseline.upper()} benefits more from streaming adaptation.",
                "Adaptation gains are effectively tied.",
            ),
            (
                "Use this as the main PCN adaptation claim."
                if gain_adv > 0
                else "Avoid claiming stronger PCN adaptation for this run."
            ),
        )

        pcn_ll = lookup.get(("pcn", "ultimate_streaming", "log_loss"), np.nan)
        base_ll = lookup.get((baseline, "ultimate_streaming", "log_loss"), np.nan)
        pcn_brier = lookup.get(("pcn", "ultimate_streaming", "brier"), np.nan)
        base_brier = lookup.get((baseline, "ultimate_streaming", "brier"), np.nan)
        pcn_auc = lookup.get(("pcn", "ultimate_streaming", "roc_auc"), np.nan)
        base_auc = lookup.get((baseline, "ultimate_streaming", "roc_auc"), np.nan)
        add(
            "Is PCN also better as a probabilistic forecaster?",
            (
                f"US LogLoss: PCN={_fmt_num(pcn_ll)}, {baseline.upper()}={_fmt_num(base_ll)}; "
                f"Brier: PCN={_fmt_num(pcn_brier)}, {baseline.upper()}={_fmt_num(base_brier)}; "
                f"ROC AUC: PCN={_fmt_num(pcn_auc)}, {baseline.upper()}={_fmt_num(base_auc)}."
            ),
            (
                "PCN is stronger on probability-sensitive metrics."
                if pcn_ll < base_ll and pcn_brier < base_brier and pcn_auc >= base_auc
                else f"{baseline.upper()} remains stronger on at least part of probabilistic forecasting."
            ),
            (
                "Present PCN as globally stronger."
                if pcn_ll < base_ll and pcn_brier < base_brier and pcn_auc >= base_auc
                else "Present PCN as a hard-label/streaming adaptation model, not necessarily best calibrated."
            ),
        )

        pcn_ece = lookup.get(("pcn", "ultimate_streaming", "ece"), np.nan)
        base_ece = lookup.get((baseline, "ultimate_streaming", "ece"), np.nan)
        add(
            "Is calibration a limitation?",
            f"US ECE: PCN={_fmt_num(pcn_ece)}, {baseline.upper()}={_fmt_num(base_ece)}. Lower is better.",
            (
                "PCN calibration is weaker in this run."
                if pcn_ece > base_ece
                else "PCN calibration is competitive in this run."
            ),
            (
                "Consider temperature scaling, Platt scaling, or isotonic regression."
                if pcn_ece > base_ece
                else "Calibration does not look like the main limitation here."
            ),
        )

    if not streaming_comparison.empty:
        row = streaming_comparison.iloc[0]
        add(
            "What is the direct PCN streaming advantage over the baseline?",
            (
                f"Accuracy advantage={_fmt_pct_delta(row.get('PCN_Streaming_Advantage_accuracy', np.nan))}; "
                f"LogLoss advantage={_fmt_num(row.get('PCN_Streaming_Advantage_log_loss', np.nan))}; "
                f"Brier advantage={_fmt_num(row.get('PCN_Streaming_Advantage_brier', np.nan))}. "
                "Positive means PCN is better."
            ),
            "This row is the direct head-to-head Ultimate Streaming comparison.",
            "Use it as the compact model-comparison table in reports.",
        )

    if not elo_rank.empty and "Accuracy Disagree Elo" in elo_rank.columns:
        pcn_row = _row_for(elo_rank, "pcn", "Ultimate Streaming")
        base_row = _row_for(elo_rank, baseline, "Ultimate Streaming") if baseline is not None else None
        if pcn_row is not None:
            pcn_disagree = pcn_row.get("Accuracy Disagree Elo", np.nan)
            base_disagree = base_row.get("Accuracy Disagree Elo", np.nan) if base_row is not None else np.nan
            add(
                "Does PCN learn beyond Elo/Rank heuristics?",
                (
                    f"Accuracy when disagreeing with Elo: PCN={_fmt_pct(pcn_disagree)}, "
                    f"{baseline.upper() if baseline else 'baseline'}={_fmt_pct(base_disagree)}."
                ),
                (
                    "PCN is stronger on Elo-disagreement cases."
                    if pd.notna(base_disagree) and pcn_disagree > base_disagree
                    else "PCN does not clearly outperform on Elo-disagreement cases."
                ),
                "Use Elo-disagreement accuracy to support or limit the temporal-pattern claim.",
            )

    if not efficiency.empty:
        add(
            "Is PCN update-efficient?",
            "Update-time columns were available, so efficiency metrics were computed.",
            "Inspect efficiency_metrics_table for gain per update time and updated parameter.",
            "Report efficiency only when update-time instrumentation is reliable.",
        )
    else:
        add(
            "Is PCN update-efficient?",
            "No update_time/update_time_ms columns were available.",
            "Efficiency cannot be concluded from this run.",
            "Add update-time logging before making compute-efficiency claims.",
        )

    return pd.DataFrame(rows)


def write_interpretation_report(summary: pd.DataFrame, output_dir: str) -> None:
    path = os.path.join(output_dir, "metrics_interpretation_report.md")
    lines = [
        "# Metrics Interpretation Report",
        "",
        "Use this file first. The CSV files are detailed evidence tables; this report turns them into research conclusions.",
        "",
        "## How To Read The New Metrics",
        "",
        "- Accuracy/F1/Balanced Accuracy answer hard-label winner prediction.",
        "- Log Loss/Brier/ROC AUC/PR AUC answer probabilistic forecasting quality.",
        "- Adaptation Gain answers whether a mode improved over Static for the same model.",
        "- Streaming Advantage answers whether PCN beats NN/ResNet in Ultimate Streaming.",
        "- Adaptation Gain Advantage answers whether PCN benefits more from streaming than NN/ResNet.",
        "- ECE and overconfident-wrong rate answer calibration and confidence reliability.",
        "- Elo/Rank disagreement accuracy answers whether the model does more than copy heuristics.",
        "",
        "## Executive Conclusions",
        "",
    ]
    if summary.empty:
        lines.append("No summary rows were generated.")
    else:
        for _, row in summary.iterrows():
            lines.extend(
                [
                    f"### {row['question']}",
                    "",
                    f"**Evidence:** {row['evidence']}",
                    "",
                    f"**Conclusion:** {row['conclusion']}",
                    "",
                    f"**Recommended claim:** {row['recommended_claim']}",
                    "",
                ]
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _row_for(df: pd.DataFrame, model: str | None, mode: str):
    if model is None:
        return None
    subset = df[(df["model"].astype(str).str.lower() == model) & (df["mode"].astype(str) == mode)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _fmt_num(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def _fmt_pct_delta(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:+.2f} pp"


def _sign_conclusion(value: float, positive: str, negative: str, tied: str, tol: float = 1e-6) -> str:
    if pd.isna(value) or abs(float(value)) <= tol:
        return tied
    return positive if value > 0 else negative


def generate_metric_tables(
    predictions: pd.DataFrame, output_dir: str, bootstrap_resamples: int = 1000
) -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    df = normalize_prediction_records(predictions)
    df.to_csv(os.path.join(output_dir, "prediction_records.csv"), index=False)

    aggregate, reliability = compute_aggregate_metrics(df)
    gains, streaming_comparison = compute_adaptation_gains(aggregate)
    preq_cols = [
        "model",
        "mode",
        "prequential_accuracy",
        "prequential_log_loss",
        "prequential_brier",
        "time_weighted_accuracy",
        "time_weighted_log_loss",
        "time_weighted_brier",
    ]
    for col in preq_cols:
        if col not in aggregate.columns:
            aggregate[col] = np.nan
    preq = aggregate[preq_cols].copy()
    preq["mode"] = preq["mode"].map(lambda m: MODE_LABELS.get(m, m))
    moving_summary, moving_curves = compute_moving_window_metrics(df)
    efficiency = compute_update_efficiency(df, aggregate)
    elo_rank = compute_elo_rank_disagreement_metrics(df)
    segments = compute_segment_metrics(df)
    adaptation_speed = compute_adaptation_speed(df)
    yearwise = compute_yearwise_metrics(df)
    ci_table = compute_bootstrap_ci_table(df, n_resamples=bootstrap_resamples)
    paired_ci = compute_paired_bootstrap_comparison_ci(df, n_resamples=bootstrap_resamples)
    executive_summary = build_executive_summary(aggregate, gains, streaming_comparison, elo_rank, efficiency)

    main_long = aggregate.copy()
    main_long["mode_label"] = main_long["mode"].map(lambda m: MODE_LABELS.get(m, m))
    main_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier",
        "ece",
        "mean_confidence",
        "overconfident_wrong_rate",
    ]
    main_table = main_long.melt(
        id_vars=["model", "mode_label"],
        value_vars=[m for m in main_metrics if m in main_long.columns],
        var_name="metric",
        value_name="value",
    ).pivot_table(index=["metric", "model"], columns="mode_label", values="value", aggfunc="first")
    main_table = main_table.reindex(columns=[MODE_LABELS[m] for m in MODE_ORDER])
    main_table = main_table.reset_index()

    tables = {
        "main_metrics_table": main_table,
        "adaptation_metrics_table": gains,
        "prequential_metrics_table": preq,
        "moving_window_summary_table": moving_summary,
        "reliability_diagram_data": reliability,
        "streaming_comparison_table": streaming_comparison,
        "bootstrap_ci_table": ci_table,
        "paired_bootstrap_comparison_ci_table": paired_ci,
        "executive_summary_table": executive_summary,
        "yearwise_static_metrics_table": yearwise,
        "segment_metrics_table": segments,
        "adaptation_speed_table": adaptation_speed,
        "moving_window_curves": moving_curves,
        "elo_rank_disagreement_table": elo_rank,
    }
    if not efficiency.empty:
        tables["efficiency_metrics_table"] = efficiency

    for name, table in tables.items():
        if table.empty:
            continue
        table.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        with open(os.path.join(output_dir, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(_to_markdown(table))
    write_interpretation_report(executive_summary, output_dir)
    return tables


def _to_markdown(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown(index=False)
    except ImportError:
        return table.to_csv(index=False)


def generate_metric_plots(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    gains = tables.get("adaptation_metrics_table", pd.DataFrame())
    if not gains.empty:
        _plot_bar(
            gains,
            "model",
            "Ultimate Streaming Adaptation Gain",
            "Ultimate Streaming Adaptation Gain by Model",
            os.path.join(output_dir, "ultimate_streaming_adaptation_gain.png"),
        )
        _plot_bar(
            gains,
            "model",
            "Relative Ultimate Streaming Gain",
            "Relative Ultimate Streaming Gain by Model",
            os.path.join(output_dir, "relative_ultimate_streaming_gain.png"),
        )

    curves = tables.get("moving_window_curves", pd.DataFrame())
    if not curves.empty:
        for mode in ["Online", "Ultimate Streaming"]:
            _plot_moving_metric(
                curves,
                mode,
                "moving_accuracy",
                f"Moving Window Accuracy - {mode}",
                os.path.join(output_dir, f"moving_accuracy_{mode.lower().replace(' ', '_')}.png"),
            )
        _plot_moving_metric(
            curves,
            None,
            "moving_log_loss",
            "Moving Window Log Loss",
            os.path.join(output_dir, "moving_log_loss_online_ultimate_streaming.png"),
            modes={"Online", "Ultimate Streaming"},
        )
        _plot_moving_metric(
            curves,
            None,
            "moving_brier",
            "Moving Window Brier Score",
            os.path.join(output_dir, "moving_brier_online_ultimate_streaming.png"),
            modes={"Online", "Ultimate Streaming"},
        )

    reliability = tables.get("reliability_diagram_data", pd.DataFrame())
    if not reliability.empty:
        _plot_reliability(
            reliability,
            os.path.join(output_dir, "reliability_ultimate_streaming.png"),
        )

    elo = tables.get("elo_rank_disagreement_table", pd.DataFrame())
    if not elo.empty and "Accuracy Disagree Elo" in elo.columns:
        subset = elo[["model", "mode", "Accuracy Disagree Elo"]].dropna()
        if not subset.empty:
            labels = subset["model"].astype(str) + " / " + subset["mode"].astype(str)
            plt.figure(figsize=(12, 5))
            plt.bar(labels, subset["Accuracy Disagree Elo"])
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Accuracy")
            plt.title("Elo-Disagreement Accuracy by Model and Mode")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "elo_disagreement_accuracy.png"), dpi=300)
            plt.close()


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, save_path: str) -> None:
    if y_col not in df.columns:
        return
    plot_df = df[[x_col, y_col]].dropna()
    if plot_df.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_moving_metric(
    curves: pd.DataFrame,
    mode: str | None,
    metric: str,
    title: str,
    save_path: str,
    modes: set[str] | None = None,
) -> None:
    subset = curves.copy()
    if mode is not None:
        subset = subset[subset["mode"] == mode]
    if modes is not None:
        subset = subset[subset["mode"].isin(modes)]
    subset = subset[subset["model"].isin(["nn", "pcn", "resnet"])]
    if subset.empty or metric not in subset.columns:
        return
    plt.figure(figsize=(12, 5))
    for (model, mode_name, window), group in subset.groupby(["model", "mode", "window_size"]):
        label = f"{model.upper()} {mode_name} W={window}"
        x = group["time"]
        plt.plot(x, group[metric], label=label, alpha=0.85)
    plt.title(title)
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_reliability(reliability: pd.DataFrame, save_path: str) -> None:
    subset = reliability[
        (reliability["mode"] == "Ultimate Streaming") & reliability["model"].isin(["nn", "pcn", "resnet"])
    ].copy()
    subset = subset.dropna(subset=["bin_confidence", "bin_accuracy"])
    if subset.empty:
        return
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
    for model, group in subset.groupby("model"):
        group = group.sort_values("bin_confidence")
        plt.plot(group["bin_confidence"], group["bin_accuracy"], marker="o", label=model.upper())
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram - Ultimate Streaming")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def print_adaptation_interpretation(tables: dict[str, pd.DataFrame]) -> None:
    aggregate = tables.get("main_metrics_table", pd.DataFrame())
    gains = tables.get("adaptation_metrics_table", pd.DataFrame())
    comparison = tables.get("streaming_comparison_table", pd.DataFrame())
    elo = tables.get("elo_rank_disagreement_table", pd.DataFrame())
    efficiency = tables.get("efficiency_metrics_table", pd.DataFrame())

    print("\n" + "=" * 50)
    print(" ADAPTATION METRICS INTERPRETATION")
    print("=" * 50)
    if not gains.empty and "pcn" in set(gains["model"]):
        pcn = gains[gains["model"] == "pcn"].iloc[0]
        if pcn.get("Ultimate Streaming Adaptation Gain", np.nan) > 0:
            print(
                "PCN achieves higher hard-label performance from Static to Ultimate Streaming, "
                "suggesting stronger temporal adaptation for winner classification."
            )
        if pcn.get("Adaptation Gain Advantage over NN", np.nan) > 0:
            print(
                "PCN obtains a larger Adaptation Gain from Static to Ultimate Streaming, "
                "supporting the claim that PCN benefits more from sequential updates."
            )
        if (
            pcn.get("Ultimate Streaming Adaptation Gain LogLoss", np.nan) < 0
            or pcn.get("Ultimate Streaming Adaptation Gain Brier", np.nan) < 0
        ):
            print(
                "PCN's probability calibration remains a limitation and may require post-hoc "
                "calibration such as temperature scaling, Platt scaling, or isotonic regression."
            )

    if not comparison.empty:
        row = comparison.iloc[0]
        ll_adv = row.get("PCN_Streaming_Advantage_log_loss", np.nan)
        brier_adv = row.get("PCN_Streaming_Advantage_brier", np.nan)
        auc_adv = row.get("PCN_Streaming_Advantage_roc_auc", np.nan)
        pr_adv = row.get("PCN_Streaming_Advantage_pr_auc", np.nan)
        if any(pd.notna(v) and v < 0 for v in [ll_adv, brier_adv, auc_adv, pr_adv]):
            print(
                "NN/ResNet remains stronger in at least some probability-sensitive metrics such "
                "as ROC AUC, PR AUC, Log Loss, or Brier Score, indicating better ranking quality "
                "and calibration for those criteria."
            )

    if not aggregate.empty:
        print(
            "Therefore, PCN should be presented as more effective for streaming hard-label "
            "adaptation only when the adaptation-gain columns are positive, not necessarily as "
            "globally superior in probabilistic forecasting."
        )
    if not elo.empty:
        print(
            "Elo/Rank disagreement metrics separate genuine temporal learning from simply "
            "copying ranking heuristics; inspect Accuracy Disagree Elo/Rank alongside reliance."
        )
    if not efficiency.empty:
        print(
            "Update-efficiency metrics report whether adaptation gains are achieved with low "
            "streaming update cost."
        )
    print("=" * 50 + "\n")

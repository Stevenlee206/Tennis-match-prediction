from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss

from src.continual_learning_benchmark.adaptation_metrics import (
    MODE_LABELS,
    compute_calibration_metrics,
)


EPS = 1e-15
ADAPTIVE_MODES = ("finetune", "online", "ultimate_streaming")


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _clip(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def _time_col(df: pd.DataFrame) -> str | None:
    return next((c for c in ["date", "timestamp", "tourney_date", "online_step"] if c in df.columns), None)


def create_player_elo_long_format(df: pd.DataFrame) -> pd.DataFrame:
    required = ["match_id", "model_name", "mode", "player_1_id", "player_2_id", "y_true", "y_prob", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _warn(f"Skipping player-centric Elo metrics: missing required columns {missing}.")
        return pd.DataFrame()

    base_meta = [
        c
        for c in ["match_id", "date", "timestamp", "tourney_date", "model_name", "mode", "surface", "tournament", "round", "is_upset", "prediction_before_update", "online_step"]
        if c in df.columns
    ]
    p1 = df[base_meta].copy()
    p2 = df[base_meta].copy()
    p1["player_id"] = df["player_1_id"].values
    p1["opponent_id"] = df["player_2_id"].values
    p1["player_win"] = df["y_true"].astype(int).values
    p1["player_win_prob"] = _clip(df["y_prob"].values)
    p1["player_pred_win"] = df["y_pred"].astype(int).values
    p2["player_id"] = df["player_2_id"].values
    p2["opponent_id"] = df["player_1_id"].values
    p2["player_win"] = 1 - df["y_true"].astype(int).values
    p2["player_win_prob"] = 1 - _clip(df["y_prob"].values)
    p2["player_pred_win"] = 1 - df["y_pred"].astype(int).values

    perspective_pairs = [
        ("elo_1", "elo_2", "player_elo"),
        ("elo_2", "elo_1", "opponent_elo"),
        ("player_1_elo_delta", "player_2_elo_delta", "player_elo_delta"),
        ("player_1_elo_group", "player_2_elo_group", "player_elo_group"),
        ("player_1_elo_quantile", "player_2_elo_quantile", "player_elo_quantile"),
        ("player_1_trend_group", "player_2_trend_group", "player_trend_group"),
    ]
    for p1_col, p2_col, out_col in perspective_pairs:
        if p1_col in df.columns and p2_col in df.columns:
            p1[out_col] = df[p1_col].values
            p2[out_col] = df[p2_col].values

    if "p_elo" in df.columns:
        p_elo = _clip(df["p_elo"].values)
        p1["player_p_elo"] = p_elo
        p2["player_p_elo"] = 1 - p_elo
    elif {"elo_1", "elo_2"}.issubset(df.columns):
        p_elo = 1.0 / (1.0 + 10.0 ** ((pd.to_numeric(df["elo_2"], errors="coerce") - pd.to_numeric(df["elo_1"], errors="coerce")) / 400.0))
        p_elo = _clip(p_elo)
        p1["player_p_elo"] = p_elo
        p2["player_p_elo"] = 1 - p_elo

    long_df = pd.concat([p1, p2], ignore_index=True)
    if {"player_elo", "opponent_elo"}.issubset(long_df.columns):
        long_df["elo_diff_player"] = long_df["player_elo"] - long_df["opponent_elo"]
    return enrich_player_elo_history(long_df)


def enrich_player_elo_history(long_df: pd.DataFrame, volatility_window: int = 5) -> pd.DataFrame:
    df = long_df.copy()
    if "player_elo" not in df.columns:
        _warn("Player Elo history unavailable: missing elo_1/elo_2.")
        return df
    time_col = _time_col(df)
    order_cols = ["model_name", "mode", "player_id"] + ([time_col] if time_col else []) + ["match_id"]
    df = df.sort_values(order_cols, kind="mergesort").reset_index(drop=True)
    grouped = df.groupby(["model_name", "mode", "player_id"], sort=False)
    derived_delta = grouped["player_elo"].diff()
    if "player_elo_delta" not in df.columns:
        df["player_elo_delta"] = derived_delta
    else:
        df["player_elo_delta"] = pd.to_numeric(df["player_elo_delta"], errors="coerce").fillna(derived_delta)
    df["abs_elo_delta"] = df["player_elo_delta"].abs()

    if "player_elo_quantile" not in df.columns:
        pct_rank = df.groupby(["model_name", "mode"])["player_elo"].rank(pct=True, method="average")
        df["player_elo_quantile"] = np.select(
            [pct_rank <= 0.25, pct_rank <= 0.50, pct_rank <= 0.75],
            ["low", "mid_low", "mid_high"],
            default="high",
        )
    if "player_elo_group" not in df.columns:
        df["player_elo_group"] = df["player_elo_quantile"].astype(str)
    if "player_trend_group" not in df.columns:
        df["player_trend_group"] = np.select(
            [df["player_elo_delta"] > 0, df["player_elo_delta"] < 0],
            ["rising", "declining"],
            default="stable",
        )

    df["elo_volatility"] = grouped["player_elo_delta"].transform(
        lambda s: s.rolling(volatility_window, min_periods=2).std()
    )
    vol_threshold = df["elo_volatility"].quantile(0.75)
    df["volatility_group"] = np.where(df["elo_volatility"] >= vol_threshold, "high_volatility", "other")
    high_delta_threshold = df["abs_elo_delta"].quantile(0.75)
    df["high_delta_group"] = np.where(df["abs_elo_delta"] >= high_delta_threshold, "high_delta", "other")
    return df


def _metrics(group: pd.DataFrame) -> dict[str, float]:
    y = group["player_win"].astype(int).values
    p = _clip(group["player_win_prob"].values)
    pred = group["player_pred_win"].astype(int).values
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ece": float(compute_calibration_metrics(y, p, pred)["ece"]),
        "n_matches": len(group),
    }


def compute_elo_conditioned_adaptation_gain(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["player_trend_group", "player_elo_group", "player_elo_quantile", "high_delta_group", "volatility_group"] if c in long_df.columns]
    if not group_cols:
        _warn("Skipping Elo-conditioned adaptation gain: no Elo/trend group.")
        return pd.DataFrame()
    rows = []
    for group_col in group_cols:
        grouped_metrics = {}
        for (model, mode, elo_group), group in long_df.groupby(["model_name", "mode", group_col], dropna=False):
            grouped_metrics[(model, mode, str(elo_group))] = _metrics(group)
        for model in long_df["model_name"].unique():
            groups = {key[2] for key in grouped_metrics if key[0] == model}
            for mode in ADAPTIVE_MODES:
                for elo_group in groups:
                    static = grouped_metrics.get((model, "static", elo_group))
                    adaptive = grouped_metrics.get((model, mode, elo_group))
                    if not static or not adaptive:
                        continue
                    rows.append(
                        {
                            "model_name": model,
                            "adaptive_mode": MODE_LABELS.get(mode, mode),
                            "group_type": group_col,
                            "elo_group": elo_group,
                            "static_accuracy": static["accuracy"],
                            "adaptive_accuracy": adaptive["accuracy"],
                            "elo_pag_accuracy": adaptive["accuracy"] - static["accuracy"],
                            "static_logloss": static["logloss"],
                            "adaptive_logloss": adaptive["logloss"],
                            "elo_pag_logloss": static["logloss"] - adaptive["logloss"],
                            "static_brier": static["brier"],
                            "adaptive_brier": adaptive["brier"],
                            "elo_pag_brier": static["brier"] - adaptive["brier"],
                            "n_matches": adaptive["n_matches"],
                        }
                    )
    return pd.DataFrame(rows)


def compute_elo_residual_gain(long_df: pd.DataFrame) -> pd.DataFrame:
    if "player_p_elo" not in long_df.columns:
        _warn("Skipping Elo residual gain: missing p_elo.")
        return pd.DataFrame()
    rows = []
    grouping_schemes = [("global", [])]
    grouping_schemes += [(c, [c]) for c in ["player_elo_group", "player_trend_group", "high_delta_group", "volatility_group"] if c in long_df.columns]
    for group_type, extra_cols in grouping_schemes:
        group_cols = ["model_name", "mode"] + extra_cols
        for keys, group in long_df.groupby(group_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(group_cols, keys))
            row["group_type"] = group_type
            row["elo_group"] = row.get(extra_cols[0], "global") if extra_cols else "global"
            row["trend_group"] = row.get("player_trend_group", "all")
            y = group["player_win"].astype(int).values
            p_model = _clip(group["player_win_prob"].values)
            p_elo = _clip(group["player_p_elo"].values)
            ll_elo = float(log_loss(y, p_elo, labels=[0, 1]))
            ll_model = float(log_loss(y, p_model, labels=[0, 1]))
            brier_elo = float(brier_score_loss(y, p_elo))
            brier_model = float(brier_score_loss(y, p_model))
            row.update(
                {
                    "LL_Elo": ll_elo,
                    "LL_Model": ll_model,
                    "EloResidualGain_LL": ll_elo - ll_model,
                    "Brier_Elo": brier_elo,
                    "Brier_Model": brier_model,
                    "EloResidualGain_Brier": brier_elo - brier_model,
                    "n_matches": len(group),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def compute_high_elo_change_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    if "high_delta_group" not in long_df.columns:
        _warn("Skipping high Elo-change metrics: missing Elo delta.")
        return pd.DataFrame()
    rows = []
    metric_lookup = {}
    for (model, mode, delta_group), group in long_df.groupby(["model_name", "mode", "high_delta_group"]):
        metric_lookup[(model, mode, delta_group)] = _metrics(group)
    for (model, mode, delta_group), values in metric_lookup.items():
        static = metric_lookup.get((model, "static", delta_group))
        rows.append(
            {
                "model_name": model,
                "mode": MODE_LABELS.get(mode, mode),
                "high_delta_group": delta_group,
                "accuracy": values["accuracy"],
                "logloss": values["logloss"],
                "brier": values["brier"],
                "ece": values["ece"],
                "adaptation_gain_accuracy": values["accuracy"] - static["accuracy"] if static else np.nan,
                "relative_adaptation_gain_accuracy": (values["accuracy"] - static["accuracy"]) / static["accuracy"] * 100 if static and static["accuracy"] else np.nan,
                "adaptation_gain_logloss": static["logloss"] - values["logloss"] if static else np.nan,
                "adaptation_gain_brier": static["brier"] - values["brier"] if static else np.nan,
                "n_matches": values["n_matches"],
            }
        )
    return pd.DataFrame(rows)


def compute_elo_disagreement_adaptation(long_df: pd.DataFrame) -> pd.DataFrame:
    if "player_p_elo" not in long_df.columns:
        _warn("Skipping Elo disagreement adaptation: missing p_elo/elo_pred.")
        return pd.DataFrame()
    df = long_df.copy()
    df["player_elo_pred"] = (df["player_p_elo"] >= 0.5).astype(int)
    df["model_disagree_elo"] = df["player_pred_win"] != df["player_elo_pred"]
    rows = []
    grouping_schemes = [("global", [])]
    grouping_schemes += [(c, [c]) for c in ["player_elo_group", "player_trend_group", "high_delta_group"] if c in df.columns]
    for group_type, extra_cols in grouping_schemes:
        group_cols = ["model_name", "mode"] + extra_cols
        raw = {}
        for keys, group in df.groupby(group_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            key = tuple(keys)
            disagree = group[group["model_disagree_elo"]]
            agree = group[~group["model_disagree_elo"]]
            metrics = {
                "elo_disagreement_rate": float(group["model_disagree_elo"].mean()),
                "acc_agree_elo": float((agree["player_pred_win"] == agree["player_win"]).mean()) if len(agree) else np.nan,
                "acc_disagree_elo": float((disagree["player_pred_win"] == disagree["player_win"]).mean()) if len(disagree) else np.nan,
                "ll_disagree_elo": _metrics(disagree)["logloss"] if len(disagree) else np.nan,
                "brier_disagree_elo": _metrics(disagree)["brier"] if len(disagree) else np.nan,
                "n_disagree": len(disagree),
                "n_agree": len(agree),
            }
            raw[key] = metrics
        for key, values in raw.items():
            row = dict(zip(group_cols, key))
            mode = row["mode"]
            static_key = tuple("static" if col == "mode" else row[col] for col in group_cols)
            static = raw.get(static_key)
            row["group_type"] = group_type
            row["elo_group"] = row.get(extra_cols[0], "global") if extra_cols else "global"
            row["mode"] = MODE_LABELS.get(mode, mode)
            row.update(values)
            row["disagree_gain_accuracy"] = values["acc_disagree_elo"] - static["acc_disagree_elo"] if static else np.nan
            row["disagree_gain_logloss"] = static["ll_disagree_elo"] - values["ll_disagree_elo"] if static else np.nan
            row["disagree_gain_brier"] = static["brier_disagree_elo"] - values["brier_disagree_elo"] if static else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compute_elo_trend_alignment(long_df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    if "player_elo" not in long_df.columns:
        _warn("Skipping Elo trend alignment: missing player Elo history.")
        return pd.DataFrame()
    df = long_df.copy()
    time_col = _time_col(df)
    order = ["model_name", "mode", "player_id"] + ([time_col] if time_col else []) + ["match_id"]
    df = df.sort_values(order, kind="mergesort")
    grouped = df.groupby(["model_name", "mode", "player_id"], sort=False)
    df["delta_model_prob"] = grouped["player_win_prob"].diff(lag)
    df["delta_elo"] = grouped["player_elo"].diff(lag)
    valid = df[df["delta_model_prob"].notna() & df["delta_elo"].notna() & (df["delta_model_prob"] != 0) & (df["delta_elo"] != 0)]
    rows = []
    for group_col in [None, "player_trend_group", "player_elo_quantile"]:
        if group_col is not None and group_col not in valid.columns:
            continue
        cols = ["model_name", "mode"] + ([group_col] if group_col else [])
        for keys, group in valid.groupby(cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(cols, keys))
            row["mode"] = MODE_LABELS.get(row["mode"], row["mode"])
            row["group_name"] = str(row.pop(group_col)) if group_col else "global"
            row["trend_corr"] = float(group["delta_model_prob"].corr(group["delta_elo"])) if len(group) > 1 else np.nan
            row["directional_alignment"] = float((np.sign(group["delta_model_prob"]) == np.sign(group["delta_elo"])).mean())
            row["n_observations"] = len(group)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_samples_to_catch_up(long_df: pd.DataFrame, delta_acc: float = 0.005, delta_ll: float = 0.005) -> pd.DataFrame:
    if "player_trend_group" not in long_df.columns:
        _warn("Skipping samples-to-catch-up: missing trend events.")
        return pd.DataFrame()
    rows = []
    event_groups = ["rising", "declining", "high_delta"]
    for model in long_df["model_name"].unique():
        static = long_df[(long_df["model_name"] == model) & (long_df["mode"] == "static")]
        for mode in ADAPTIVE_MODES:
            adaptive = long_df[(long_df["model_name"] == model) & (long_df["mode"] == mode)]
            matched = adaptive.merge(
                static[["match_id", "player_id", "player_win", "player_win_prob", "player_pred_win"]],
                on=["match_id", "player_id"],
                suffixes=("_adaptive", "_static"),
            )
            if matched.empty:
                continue
            # The adaptive dataframe already contributes trend/high-delta columns.
            # Merging them a second time would rename them to *_x/*_y.
            for event in event_groups:
                acc_samples, ll_samples, player_count = [], [], 0
                for player, player_df in matched.groupby("player_id"):
                    time_col = _time_col(player_df)
                    sort_cols = ([time_col] if time_col else []) + ["match_id"]
                    player_df = player_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
                    if event == "high_delta" and "high_delta_group" in player_df.columns:
                        event_idx = player_df.index[player_df["high_delta_group"] == "high_delta"].tolist()
                    elif "player_trend_group" in player_df.columns:
                        event_idx = player_df.index[player_df["player_trend_group"] == event].tolist()
                    else:
                        event_idx = []
                    if not event_idx:
                        continue
                    player_count += 1
                    post = player_df.iloc[event_idx[0]:].reset_index(drop=True)
                    acc_k = np.nan
                    ll_k = np.nan
                    for k in range(1, len(post) + 1):
                        chunk = post.iloc[:k]
                        acc_gain = (
                            (chunk["player_pred_win_adaptive"] == chunk["player_win_adaptive"]).mean()
                            - (chunk["player_pred_win_static"] == chunk["player_win_static"]).mean()
                        )
                        y = chunk["player_win_adaptive"].astype(int).values
                        ll_gain = log_loss(y, _clip(chunk["player_win_prob_static"]), labels=[0, 1]) - log_loss(
                            y, _clip(chunk["player_win_prob_adaptive"]), labels=[0, 1]
                        )
                        if pd.isna(acc_k) and acc_gain >= delta_acc:
                            acc_k = k
                        if pd.isna(ll_k) and ll_gain >= delta_ll:
                            ll_k = k
                    acc_samples.append(acc_k)
                    ll_samples.append(ll_k)
                if player_count:
                    acc_valid = pd.Series(acc_samples).dropna()
                    ll_valid = pd.Series(ll_samples).dropna()
                    rows.append(
                        {
                            "model_name": model,
                            "mode": MODE_LABELS.get(mode, mode),
                            "trend_event_group": event,
                            "mean_samples_to_catch_up_acc": acc_valid.mean(),
                            "median_samples_to_catch_up_acc": acc_valid.median(),
                            "percentage_players_caught_up_acc": len(acc_valid) / player_count,
                            "mean_samples_to_catch_up_ll": ll_valid.mean(),
                            "median_samples_to_catch_up_ll": ll_valid.median(),
                            "percentage_players_caught_up_ll": len(ll_valid) / player_count,
                            "n_players": player_count,
                        }
                    )
    return pd.DataFrame(rows)


def compute_elo_volatility_robustness(long_df: pd.DataFrame) -> pd.DataFrame:
    if "volatility_group" not in long_df.columns:
        _warn("Skipping Elo volatility robustness: missing volatility history.")
        return pd.DataFrame()
    disagreement = compute_elo_disagreement_adaptation(long_df)
    rows = []
    metrics_lookup = {}
    for (model, mode, vol_group), group in long_df.groupby(["model_name", "mode", "volatility_group"]):
        metrics_lookup[(model, mode, vol_group)] = _metrics(group)
    for (model, mode, vol_group), values in metrics_lookup.items():
        static = metrics_lookup.get((model, "static", vol_group))
        disagree_subset = disagreement[(disagreement["model_name"] == model) & (disagreement["mode"] == MODE_LABELS.get(mode, mode))] if not disagreement.empty else pd.DataFrame()
        residual_gain = np.nan
        group = long_df[(long_df["model_name"] == model) & (long_df["mode"] == mode) & (long_df["volatility_group"] == vol_group)]
        if "player_p_elo" in group.columns and not group.empty:
            y = group["player_win"].astype(int).values
            residual_gain = log_loss(y, _clip(group["player_p_elo"]), labels=[0, 1]) - log_loss(
                y, _clip(group["player_win_prob"]), labels=[0, 1]
            )
        rows.append(
            {
                "model_name": model,
                "mode": MODE_LABELS.get(mode, mode),
                "volatility_group": vol_group,
                "accuracy": values["accuracy"],
                "logloss": values["logloss"],
                "brier": values["brier"],
                "adaptation_gain_accuracy": values["accuracy"] - static["accuracy"] if static else np.nan,
                "adaptation_gain_logloss": static["logloss"] - values["logloss"] if static else np.nan,
                "elo_residual_gain_logloss": residual_gain,
                "acc_disagree_elo": disagree_subset["acc_disagree_elo"].mean() if not disagree_subset.empty else np.nan,
                "n_matches": values["n_matches"],
            }
        )
    return pd.DataFrame(rows)


def compute_pcn_elo_continual_advantage(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    specs = [
        ("elo_conditioned_adaptation_gain_table", "elo_pag_accuracy", True, ["adaptive_mode", "group_type", "elo_group"]),
        ("elo_conditioned_adaptation_gain_table", "elo_pag_logloss", True, ["adaptive_mode", "group_type", "elo_group"]),
        ("elo_residual_gain_table", "EloResidualGain_LL", True, ["mode", "group_type", "elo_group"]),
        ("high_elo_change_metrics_table", "accuracy", True, ["mode", "high_delta_group"]),
        ("high_elo_change_metrics_table", "adaptation_gain_accuracy", True, ["mode", "high_delta_group"]),
        ("elo_disagreement_adaptation_table", "disagree_gain_accuracy", True, ["mode", "group_type", "elo_group"]),
        ("elo_trend_alignment_table", "trend_corr", True, ["mode", "group_name"]),
        ("elo_trend_alignment_table", "directional_alignment", True, ["mode", "group_name"]),
        ("samples_to_catch_up_elo_table", "mean_samples_to_catch_up_acc", False, ["mode", "trend_event_group"]),
        ("elo_volatility_robustness_table", "adaptation_gain_accuracy", True, ["mode", "volatility_group"]),
    ]
    rows = []
    for table_name, metric, higher, group_cols in specs:
        table = tables.get(table_name, pd.DataFrame())
        if table.empty or metric not in table.columns:
            continue
        available_groups = [c for c in group_cols if c in table.columns]
        for keys, group in table.groupby(available_groups, dropna=False) if available_groups else [((), table)]:
            keys = keys if isinstance(keys, tuple) else (keys,)
            pcn = group[group["model_name"].astype(str).str.lower() == "pcn"][metric]
            nn = group[group["model_name"].astype(str).str.lower().isin(["nn", "resnet"])][metric]
            if pcn.empty or nn.empty:
                continue
            pcn_value, nn_value = float(pcn.mean()), float(nn.mean())
            rows.append(
                {
                    "metric": metric,
                    "group": " / ".join(map(str, keys)),
                    "mode": next((str(group.iloc[0][c]) for c in ["mode", "adaptive_mode"] if c in group.columns), "NA"),
                    "PCN value": pcn_value,
                    "NN value": nn_value,
                    "PCN advantage": pcn_value - nn_value if higher else nn_value - pcn_value,
                }
            )
    return pd.DataFrame(rows)


def generate_player_elo_tables(predictions: pd.DataFrame, output_dir: str) -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    write_player_elo_metrics_guide(output_dir)
    try:
        long_df = create_player_elo_long_format(predictions)
    except Exception as exc:
        _warn(f"Skipping all player-centric Elo metrics: failed to create long format: {exc}")
        return {}
    if long_df.empty:
        return {}

    def safe_compute(name, fn):
        try:
            return fn(long_df)
        except Exception as exc:
            _warn(f"Skipping {name}: {type(exc).__name__}: {exc}")
            return pd.DataFrame()

    tables = {
        "player_elo_long_format": long_df,
        "elo_conditioned_adaptation_gain_table": safe_compute(
            "Elo-conditioned adaptation gain", compute_elo_conditioned_adaptation_gain
        ),
        "elo_residual_gain_table": safe_compute("Elo residual gain", compute_elo_residual_gain),
        "high_elo_change_metrics_table": safe_compute("high Elo-change metrics", compute_high_elo_change_metrics),
        "elo_disagreement_adaptation_table": safe_compute(
            "Elo disagreement adaptation", compute_elo_disagreement_adaptation
        ),
        "elo_trend_alignment_table": safe_compute("Elo trend alignment", compute_elo_trend_alignment),
        "samples_to_catch_up_elo_table": safe_compute("samples-to-catch-up", compute_samples_to_catch_up),
        "elo_volatility_robustness_table": safe_compute(
            "Elo volatility robustness", compute_elo_volatility_robustness
        ),
    }
    try:
        tables["pcn_elo_continual_advantage_table"] = compute_pcn_elo_continual_advantage(tables)
    except Exception as exc:
        _warn(f"Skipping PCN Elo continual advantage table: {type(exc).__name__}: {exc}")
        tables["pcn_elo_continual_advantage_table"] = pd.DataFrame()
    for name, table in tables.items():
        if table.empty:
            continue
        table.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        with open(os.path.join(output_dir, f"{name}.md"), "w", encoding="utf-8") as f:
            try:
                f.write(table.to_markdown(index=False))
            except ImportError:
                f.write(table.to_csv(index=False))
    try:
        generate_player_elo_plots(tables, output_dir)
    except Exception as exc:
        _warn(f"Skipping player-centric Elo plots: {type(exc).__name__}: {exc}")
    try:
        print_player_elo_interpretation(tables, output_dir)
    except Exception as exc:
        _warn(f"Skipping player-centric Elo interpretation: {type(exc).__name__}: {exc}")
    return tables


def write_player_elo_metrics_guide(output_dir: str) -> None:
    guide = """# Player-Centric Elo Adaptation Metrics Guide

Read this file before interpreting the CSV tables in this directory.

## Research Question

These metrics test whether PCN learns changing player dynamics better than NN/ResNet during Finetune, Online, and Ultimate Streaming updates.

High Elo reliance alone is not evidence of continual adaptation. Strong evidence requires improvement for rising, declining, high-Elo-change, volatile, or Elo-disagreement cases.

## Direction Convention

- Higher is better: Accuracy, F1, Elo Residual Gain, Adaptation Gain, trend correlation, directional alignment, PCN advantage.
- Lower is better: Log Loss, Brier Score, ECE, samples-to-catch-up.
- Gain values are normalized so positive generally means improvement.
- In `pcn_elo_continual_advantage_table`, positive always means PCN is better.

## Player Elo Long Format

File: `player_elo_long_format.csv`

Each original match becomes two rows, one from each player's perspective.

- `player_win`: whether the current player won.
- `player_win_prob`: model probability that the current player wins.
- `player_pred_win`: model hard-label prediction for the current player.
- `player_elo`, `opponent_elo`: pre-match Elo ratings.
- `elo_diff_player`: player Elo minus opponent Elo.
- `player_p_elo`: Elo baseline probability that the current player wins.
- `player_elo_delta`: change in the player's Elo since their previous match.
- `abs_elo_delta`: magnitude of the Elo change.
- `player_elo_quantile`: low, mid-low, mid-high, or high Elo group.
- `player_trend_group`: rising, declining, or stable based on Elo change direction.
- `high_delta_group`: whether the Elo change is in the top 25%.
- `volatility_group`: whether recent Elo-change volatility is in the top 25%.

This is a diagnostic dataset. It should not be used as model training input.

## Elo-Conditioned Player Adaptation Gain

File: `elo_conditioned_adaptation_gain_table.csv`

Measures whether an adaptive mode improves over Static inside each Elo-related player group.

- `static_accuracy`: Static accuracy in the group.
- `adaptive_accuracy`: Finetune, Online, or Ultimate Streaming accuracy.
- `elo_pag_accuracy = adaptive_accuracy - static_accuracy`.
- `elo_pag_logloss = static_logloss - adaptive_logloss`.
- `elo_pag_brier = static_brier - adaptive_brier`.
- `group_type`: which grouping produced the row, such as trend, Elo quantile, high delta, or volatility.
- `elo_group`: the specific group value.

Interpretation:

- Positive gain means the adaptive mode improved over Static.
- PCN has stronger player-centric adaptation when its gains are larger than NN's, especially for rising, declining, high-delta, and high-volatility groups.

## Elo Residual Gain

File: `elo_residual_gain_table.csv`

Measures whether the model predicts better probabilities than the Elo baseline.

- `LL_Elo`: Elo baseline Log Loss.
- `LL_Model`: model Log Loss.
- `EloResidualGain_LL = LL_Elo - LL_Model`.
- `Brier_Elo`: Elo baseline Brier Score.
- `Brier_Model`: model Brier Score.
- `EloResidualGain_Brier = Brier_Elo - Brier_Model`.

Interpretation:

- Positive residual gain means the model improves on Elo.
- Negative residual gain means Elo alone is a better probabilistic predictor.
- Compare PCN and NN within the same mode and group.

## High Elo-Change Metrics

File: `high_elo_change_metrics_table.csv`

Evaluates matches involving players whose absolute Elo change is at or above the 75th percentile.

- `accuracy`, `logloss`, `brier`, `ece`: performance inside the group.
- `adaptation_gain_accuracy`: mode accuracy minus Static accuracy.
- `relative_adaptation_gain_accuracy`: percentage improvement relative to Static accuracy.
- `adaptation_gain_logloss`: Static Log Loss minus mode Log Loss.
- `adaptation_gain_brier`: Static Brier minus mode Brier.

Interpretation:

- Positive adaptation gains indicate successful adaptation to rapidly changing players.
- Lower ECE indicates more reliable confidence.

## Elo Disagreement Adaptation

File: `elo_disagreement_adaptation_table.csv`

Tests whether the model learns when to deviate from Elo.

- `elo_disagreement_rate`: fraction of predictions that disagree with Elo.
- `acc_agree_elo`: accuracy when model and Elo agree.
- `acc_disagree_elo`: accuracy when model and Elo disagree.
- `ll_disagree_elo`, `brier_disagree_elo`: probability quality on disagreement cases.
- `disagree_gain_accuracy`: adaptive disagreement accuracy minus Static disagreement accuracy.
- `disagree_gain_logloss`: Static disagreement Log Loss minus adaptive disagreement Log Loss.
- `disagree_gain_brier`: Static disagreement Brier minus adaptive disagreement Brier.

Interpretation:

- High disagreement rate alone is not good.
- Strong evidence requires positive disagreement gain and good `acc_disagree_elo`.

## Elo Trend Alignment

File: `elo_trend_alignment_table.csv`

Checks whether changes in model win probability follow changes in player Elo.

- `trend_corr`: correlation between probability change and Elo change.
- `directional_alignment`: fraction where probability and Elo move in the same direction.
- `n_observations`: number of valid change pairs.

Interpretation:

- Higher values mean model probabilities track Elo dynamics more closely.
- This does not automatically mean better match prediction; inspect it together with Accuracy and Elo Residual Gain.

## Samples-to-Catch-Up

File: `samples_to_catch_up_elo_table.csv`

Measures how many subsequent player matches an adaptive mode needs to outperform Static after a rising, declining, or high-delta event.

- `mean_samples_to_catch_up_acc`, `median_samples_to_catch_up_acc`: matches needed to achieve the Accuracy gain threshold.
- `percentage_players_caught_up_acc`: fraction of players that reached the threshold.
- Log Loss columns provide the equivalent probabilistic catch-up measurement.

Interpretation:

- Lower samples-to-catch-up is better.
- Higher percentage caught up is better.
- A low average based on very few players should not be treated as strong evidence.

## Elo Volatility Robustness

File: `elo_volatility_robustness_table.csv`

Evaluates robustness for players with unstable recent Elo changes.

- `accuracy`, `logloss`, `brier`: performance in each volatility group.
- `adaptation_gain_accuracy`, `adaptation_gain_logloss`: improvement over Static.
- `elo_residual_gain_logloss`: improvement over Elo baseline.
- `acc_disagree_elo`: accuracy when disagreeing with Elo.

Interpretation:

- PCN is stronger for volatile players only when it has positive adaptation gain and compares favorably against both NN and Elo.

## PCN vs NN Continual Advantage

File: `pcn_elo_continual_advantage_table.csv`

This is the primary compact comparison table.

- `PCN value`: PCN metric value.
- `NN value`: NN/ResNet metric value.
- `PCN advantage`: direction-normalized difference.

Interpretation:

- Positive `PCN advantage` always means PCN is better.
- Negative means NN/ResNet is better.
- Use group and mode columns to identify where the advantage occurs.

## Recommended Conclusion Logic

Claim stronger PCN player-centric continual adaptation only when several of these are true:

1. PCN has larger Elo-conditioned adaptation gains than NN.
2. PCN has positive and larger Elo Residual Gain.
3. PCN improves on high Elo-change or high-volatility players.
4. PCN improves Accuracy when disagreeing with Elo.
5. PCN tracks player Elo trends better.
6. PCN needs fewer samples to catch up after player trend changes.

If PCN improves hard-label Accuracy but NN has lower Log Loss, Brier, or ECE, conclude that PCN is stronger for hard-label continual adaptation while NN remains better calibrated.
"""
    with open(os.path.join(output_dir, "README_metrics_guide.md"), "w", encoding="utf-8") as f:
        f.write(guide)


def generate_player_elo_plots(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    plot_specs = [
        ("elo_conditioned_adaptation_gain_table", "elo_group", "elo_pag_accuracy", "adaptive_mode", "elo_conditioned_adaptation_gain.png"),
        ("elo_residual_gain_table", "model_name", "EloResidualGain_LL", "mode", "elo_residual_gain.png"),
        ("high_elo_change_metrics_table", "model_name", "adaptation_gain_accuracy", "mode", "high_elo_change_adaptation_gain.png"),
        ("elo_disagreement_adaptation_table", "model_name", "disagree_gain_accuracy", "mode", "elo_disagreement_gain.png"),
        ("samples_to_catch_up_elo_table", "trend_event_group", "mean_samples_to_catch_up_acc", "model_name", "samples_to_catch_up.png"),
    ]
    for table_name, x_col, y_col, hue_col, filename in plot_specs:
        table = tables.get(table_name, pd.DataFrame())
        if table.empty or any(c not in table.columns for c in [x_col, y_col, hue_col]):
            continue
        plot_df = table.dropna(subset=[y_col])
        if plot_df.empty:
            continue
        pivot = plot_df.pivot_table(index=x_col, columns=hue_col, values=y_col, aggfunc="mean")
        pivot.plot(kind="bar", figsize=(11, 5))
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(y_col.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()


def print_player_elo_interpretation(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    advantage = tables.get("pcn_elo_continual_advantage_table", pd.DataFrame())
    lines = [
        "PCN PLAYER-CENTRIC ELO ADAPTATION INTERPRETATION",
        "High Elo reliance alone is not sufficient evidence. Positive PCN advantage means PCN is better after metric direction is normalized.",
    ]
    if advantage.empty:
        lines.append("Insufficient matched NN/PCN Elo metrics for a direct conclusion.")
    else:
        positive = advantage[advantage["PCN advantage"] > 0]
        negative = advantage[advantage["PCN advantage"] < 0]
        lines.append(f"PCN advantage is positive for {len(positive)}/{len(advantage)} matched Elo-centric comparisons.")
        lines.append(_answer_advantage_question(advantage, "elo_pag_accuracy", "1. Static-to-Online/Ultimate gain in Elo-based player groups"))
        lines.append(_answer_advantage_question(advantage, "EloResidualGain_LL", "2. Outperforming Elo baseline in rising/declining/volatile players"))
        lines.append(_answer_advantage_question(advantage, "adaptation_gain_accuracy", "3. High Elo-change player performance"))
        lines.append(_answer_advantage_question(advantage, "disagree_gain_accuracy", "4. Improvement when disagreeing with Elo"))
        trend_subset = advantage[advantage["metric"].isin(["trend_corr", "directional_alignment"])]
        if trend_subset.empty:
            lines.append("5. Elo trend tracking: not enough trend-alignment comparisons were available.")
        else:
            lines.append(
                "5. Elo trend tracking: "
                + ("PCN tracks Elo trends better in most available comparisons." if (trend_subset["PCN advantage"] > 0).mean() >= 0.5 else "NN/ResNet tracks Elo trends better in most available comparisons.")
            )
        catchup_subset = advantage[advantage["metric"] == "mean_samples_to_catch_up_acc"]
        if catchup_subset.empty:
            lines.append("6. Samples-to-catch-up: not enough player transition events were available.")
        else:
            lines.append(
                "6. Samples-to-catch-up: "
                + ("PCN catches up faster in most available groups." if (catchup_subset["PCN advantage"] > 0).mean() >= 0.5 else "NN/ResNet catches up faster in most available groups.")
            )
        if len(positive):
            lines.append("PCN shows stronger player-centric continual adaptation only for the positive groups above.")
        if len(negative):
            lines.append("NN/ResNet remains stronger in some Elo-centric or probability-sensitive comparisons; report these limitations.")
    lines.append("PCN shows stronger player-centric continual adaptation if it achieves larger Elo-conditioned adaptation gains, higher Elo residual gain, and lower samples-to-catch-up than NN/ResNet.")
    lines.append("Lower Log Loss, Brier Score, and ECE still indicate better probability calibration; do not claim global superiority from hard-label gains alone.")
    text = "\n".join(lines)
    print("\n" + text + "\n")
    with open(os.path.join(output_dir, "player_elo_interpretation.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def _answer_advantage_question(advantage: pd.DataFrame, metric: str, question: str) -> str:
    subset = advantage[advantage["metric"] == metric]
    if subset.empty:
        return f"{question}: not enough matched PCN vs NN/ResNet data."
    positive_rate = float((subset["PCN advantage"] > 0).mean())
    mean_adv = float(subset["PCN advantage"].mean())
    if positive_rate >= 0.5:
        return f"{question}: PCN is better in {positive_rate:.0%} of matched groups; mean normalized advantage={mean_adv:.4f}."
    return f"{question}: NN/ResNet is better in most matched groups; PCN mean normalized advantage={mean_adv:.4f}."

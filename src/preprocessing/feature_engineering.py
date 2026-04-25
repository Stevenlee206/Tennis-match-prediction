import pandas as pd
import numpy as np
from .load_data import load_data
from ..config.data_config import YEARS, SURFACES


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # =========================
    # 1. Strength
    # =========================

    data["rank_diff"] = (
        data["winner_rank"] - data["loser_rank"]
    )

    data["rank_points_diff"] = (
        data["winner_rank_points"] - data["loser_rank_points"]
    )

    # =========================
    # 2. Physical
    # =========================

    data["age_diff"] = (
        data["winner_age"] - data["loser_age"]
    )

    data["height_diff"] = (
        data["winner_ht"] - data["loser_ht"]
    )

    # =========================
    # 3. Hand matchup
    # =========================

    if {"winner_hand", "loser_hand"}.issubset(data.columns):
        data["same_hand_flag"] = (
            data["winner_hand"] == data["loser_hand"]
        ).astype(int)

    # =========================
    # 4. Time Features
    # giữ nguyên year/month + thêm sin/cos encoding
    # =========================

    if "tourney_date" in data.columns:
        data["tourney_date"] = pd.to_datetime(
            data["tourney_date"],
            format="%Y%m%d",
            errors="coerce"
        )

        # giữ nguyên year/month
        data["year"] = data["tourney_date"].dt.year
        data["month"] = data["tourney_date"].dt.month
        data["day_of_year"] = data["tourney_date"].dt.dayofyear

        # cyclical encoding cho month
        data["month_sin"] = np.sin(
            2 * np.pi * data["month"] / 12
        )

        data["month_cos"] = np.cos(
            2 * np.pi * data["month"] / 12
        )

        # cyclical encoding cho day_of_year
        data["day_sin"] = np.sin(
            2 * np.pi * data["day_of_year"] / 365
        )

        data["day_cos"] = np.cos(
            2 * np.pi * data["day_of_year"] / 365
        )

    # =========================
    # 5. Interaction
    # =========================

    data["rank_age_interaction"] = (
        data["rank_diff"] * data["age_diff"]
    )

    return data


def build_elo_feature(
    df: pd.DataFrame,
    k_global: int = 32,
    k_surface: int = 24,   
) -> pd.DataFrame:

    # =========================
    # 1. Load pre-data (5 năm warm-up)
    # =========================
    dfs = []
    for year in range(YEARS[0] - 5, YEARS[0]):
        df_year = load_data(f"atp_matches_{year}.csv")
        dfs.append(df_year)
    pre_data = pd.concat(dfs, ignore_index=True).sort_values('tourney_date')

    # =========================
    # 2. Init rating dicts
    # =========================
    elo_global: dict  = {}
    elo_surface: dict = {}   

    def get_global(pid):
        return elo_global.get(pid, 1500)

    def get_surface(pid, surface):
        return elo_surface.get((pid, surface), get_global(pid))

    def update(pid_w, pid_l, surface):
        rw, rl = get_global(pid_w), get_global(pid_l)
        exp_w  = 1 / (1 + 10 ** ((rl - rw) / 400))
        elo_global[pid_w] = rw + k_global * (1 - exp_w)
        elo_global[pid_l] = rl + k_global * (0 - (1 - exp_w))

        # --- surface ---
        if surface in SURFACES:
            sw = get_surface(pid_w, surface)
            sl = get_surface(pid_l, surface)
            exp_sw = 1 / (1 + 10 ** ((sl - sw) / 400))
            elo_surface[(pid_w, surface)] = sw + k_surface * (1 - exp_sw)
            elo_surface[(pid_l, surface)] = sl + k_surface * (0 - (1 - exp_sw))

    # =========================
    # 3. Warm-up từ pre_data
    # =========================
    for _, row in pre_data.iterrows():
        update(row['winner_id'], row['loser_id'],
               row.get('surface', 'Unknown'))

    # =========================
    # 4. Build features trên df chính
    # =========================
    data = df.sort_values('tourney_date').reset_index(drop=True).copy()

    records = {
        'winner_elo':      [],
        'loser_elo':       [],
        'elo_diff':        [],
        'elo_hard_diff':   [],
        'elo_clay_diff':   [],
        'elo_grass_diff':  [],
    }

    for _, row in data.iterrows():
        w, l   = row['winner_id'], row['loser_id']
        surface = row.get('surface', 'Unknown')

        rw_g = get_global(w)
        rl_g = get_global(l)

        records['winner_elo'].append(rw_g)
        records['loser_elo'].append(rl_g)
        records['elo_diff'].append(rw_g - rl_g)

        for s in SURFACES:
            col = f'elo_{s.lower()}_diff'
            records[col].append(get_surface(w, s) - get_surface(l, s))

        update(w, l, surface)

    for col, values in records.items():
        data[col] = values

    return data

def build_recent_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    data = df.sort_values('tourney_date').copy()
    win_history = {}
    form_diff_list = []
    def get_form(player_id):
        history = win_history.get(player_id, [])
        if not history:
            return 0.5
        return sum(history[-window:]) / min(len(history), window)

    for _, row in data.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']

        form_w = get_form(winner)
        form_l = get_form(loser)

        form_diff_list.append(form_w - form_l)

        win_history.setdefault(winner, []).append(1)
        win_history.setdefault(loser, []).append(0)

    data['form_diff'] = form_diff_list
    return data
import pandas as pd
import numpy as np

def create_target(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    data = df.copy()

    np.random.seed(random_state)
    p1_is_winner = np.random.randint(0, 2, size=len(data)).astype(bool)
    sign = np.where(p1_is_winner, 1, -1)

    # =========================
    # 1. Diff features — Retains your Gao, Fatigue, and Clutch metrics
    # =========================
    diff_cols = [
        ('rank_diff',        'winner_rank',         'loser_rank'),
        ('rank_points_diff', 'winner_rank_points',  'loser_rank_points'),
        ('age_diff',         'winner_age',           'loser_age'),
        ('height_diff',      'winner_ht',            'loser_ht'),
        # --- GAO SERVE METRICS ---
        ('ace_vs_df_diff',    'w_AceVsDf',           'l_AceVsDf'),
        ('first_in_diff',     'w_FirstIn1stServe',   'l_FirstIn1stServe'),
        ('first_won_diff',    'w_FirstWonFirstIn',   'l_FirstWonFirstIn'),
        ('second_won_diff',   'w_SecondWonSecondIn', 'l_SecondWonSecondIn'),
        # --- CONTEXT METRICS ---
        ('fatigue_diff', 'w_cum_minutes', 'l_cum_minutes'),
        ('clutch_diff', 'w_ClutchFactor', 'l_ClutchFactor')
    ]

    for new_col, w_col, l_col in diff_cols:
        if w_col in data.columns and l_col in data.columns:
            data[new_col] = (data[w_col] - data[l_col]) * sign

    # =========================
    # 2. ELO + GLICKO2 + form diffs (Merged from teammate's code)
    # =========================
    elo_form_cols = [
        'elo_diff', 'elo_hard_diff', 'elo_clay_diff', 'elo_grass_diff', 
        'glicko2_diff', 'glicko2_hard_diff', 'glicko2_clay_diff', 'glicko2_grass_diff', 
        'form_diff'
    ]

    for col in elo_form_cols:
        if col in data.columns:
            data[col] = data[col] * sign

    # =========================
    # 3. Same hand flag
    # =========================
    if {'winner_hand', 'loser_hand'}.issubset(data.columns):
        p1_hand = np.where(p1_is_winner, data['winner_hand'], data['loser_hand'])
        p2_hand = np.where(p1_is_winner, data['loser_hand'],  data['winner_hand'])
        data['same_hand_flag'] = (p1_hand == p2_hand).astype(int)

    # =========================
    # 4. Target & Cleanup
    # =========================
    data['target'] = p1_is_winner.astype(int)
    
    # Drop the raw w_ and l_ columns to prevent data leakage
    raw_cols_to_drop = [
        'w_AceVsDf', 'l_AceVsDf', 'w_FirstIn1stServe', 'l_FirstIn1stServe', 
        'w_FirstWonFirstIn', 'l_FirstWonFirstIn', 'w_SecondWonSecondIn', 'l_SecondWonSecondIn',
        'w_cum_minutes', 'l_cum_minutes', 'w_ClutchFactor', 'l_ClutchFactor',
        # ---> DROP THE LEAKY GLICKO COLUMNS <---
        'winner_glicko2', 'loser_glicko2' 
    ]
    data = data.drop(columns=[c for c in raw_cols_to_drop if c in data.columns])

    return data
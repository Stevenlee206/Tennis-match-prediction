import pandas as pd
import numpy as np

def create_target(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Random hóa player1/player2 và tạo target.
    Chạy SAU build_elo_feature() và build_recent_form().
    
    Input:  df còn nguyên cột winner_*/loser_* + elo_diff, form_diff, v.v.
    Output: df với diff features đã đảo dấu đúng chiều + cột target
    """
    data = df.copy()

    np.random.seed(random_state)
    p1_is_winner = np.random.randint(0, 2, size=len(data)).astype(bool)
    sign = np.where(p1_is_winner, 1, -1)

    # =========================
    # 1. Diff features — đảo dấu theo chiều player1
    # =========================
    diff_cols = [
        ('rank_diff',        'winner_rank',         'loser_rank'),
        ('rank_points_diff', 'winner_rank_points',  'loser_rank_points'),
        ('age_diff',         'winner_age',           'loser_age'),
        ('height_diff',      'winner_ht',            'loser_ht'),
    ]

    for new_col, w_col, l_col in diff_cols:
        if w_col in data.columns and l_col in data.columns:
            data[new_col] = (data[w_col] - data[l_col]) * sign

    # =========================
    # 2. ELO + GLICKO2 + form diffs — đã là winner - loser, chỉ cần đảo dấu
    # =========================
    elo_form_cols = [
        'elo_diff',
        'elo_hard_diff',
        'elo_clay_diff',
        'elo_grass_diff',
        'glicko2_diff',
        'glicko2_hard_diff',
        'glicko2_clay_diff',
        'glicko2_grass_diff',
        'form_diff',
    ]

    for col in elo_form_cols:
        if col in data.columns:
            data[col] = data[col] * sign

    # =========================
    # 3. Same hand flag — swap cả hai phía rồi so sánh
    # =========================
    if {'winner_hand', 'loser_hand'}.issubset(data.columns):
        p1_hand = np.where(p1_is_winner, data['winner_hand'], data['loser_hand'])
        p2_hand = np.where(p1_is_winner, data['loser_hand'],  data['winner_hand'])
        data['same_hand_flag'] = (p1_hand == p2_hand).astype(int)

    # =========================
    # 4. Target
    # =========================
    data['target'] = p1_is_winner.astype(int)

    return data
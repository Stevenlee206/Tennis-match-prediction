import pandas as pd
import numpy as np

def create_target(df: pd.DataFrame, random_state: int = 42, augment: bool = True) -> pd.DataFrame:
    data = df.copy()

    if augment:
        original = data.copy()
        original['is_augmented'] = 0
        
        duplicate = data.copy()
        duplicate['is_augmented'] = 1
        data = pd.concat([original, duplicate]).sort_index(kind='stable').reset_index(drop=True)
        np.random.seed(random_state)
        num_matches = len(data) // 2
        base_signs = np.random.choice([1, -1], size=num_matches)
        sign = np.empty(len(data), dtype=int)
        sign[0::2] = base_signs       # Originals get the random sign
        sign[1::2] = -base_signs      # Duplicates get the flipped sign
        
        p1_is_winner = (sign == 1)
    else:
        data['is_augmented'] = 0
        np.random.seed(random_state)
        p1_is_winner = np.random.randint(0, 2, size=len(data)).astype(bool)
        sign = np.where(p1_is_winner, 1, -1)
    # 1. Diff features (Needs W and L columns)
    diff_cols = [
        ('rank_diff',        'winner_rank',         'loser_rank'),
        ('rank_points_diff', 'winner_rank_points',  'loser_rank_points'),
        ('age_diff',         'winner_age',           'loser_age'),
        ('height_diff',      'winner_ht',            'loser_ht'),
        # --- GAO & CONTEXT METRICS ---
        ('ace_vs_df_diff',    'w_AceVsDf',           'l_AceVsDf'),
        ('first_in_diff',     'w_FirstIn1stServe',   'l_FirstIn1stServe'),
        ('first_won_diff',    'w_FirstWonFirstIn',   'l_FirstWonFirstIn'),
        ('second_won_diff',   'w_SecondWonSecondIn', 'l_SecondWonSecondIn'),
        ('fatigue_diff', 'w_cum_minutes', 'l_cum_minutes'),
        ('clutch_diff', 'w_ClutchFactor', 'l_ClutchFactor'),
    ]

    for new_col, w_col, l_col in diff_cols:
        if w_col in data.columns and l_col in data.columns:
            data[new_col] = (data[w_col] - data[l_col]) * sign

    elo_form_matchup_cols = [
        'elo_diff', 'elo_hard_diff', 'elo_clay_diff', 'elo_grass_diff', 
        'glicko2_diff', 'glicko2_hard_diff', 'glicko2_clay_diff', 'glicko2_grass_diff', 
        'form_diff',
        'h2h_advantage_diff', 'hand_win_pct_diff',
    ]

    for col in elo_form_matchup_cols:
        if col in data.columns:
            data[col] = data[col] * sign

    # Same hand flag
    if {'winner_hand', 'loser_hand'}.issubset(data.columns):
        p1_hand = np.where(p1_is_winner, data['winner_hand'], data['loser_hand'])
        p2_hand = np.where(p1_is_winner, data['loser_hand'],  data['winner_hand'])
        data['same_hand_flag'] = (p1_hand == p2_hand).astype(int)

    # Target & Cleanup
    data['target'] = p1_is_winner.astype(int)
    
    raw_cols_to_drop = [
        'w_AceVsDf', 'l_AceVsDf', 'w_FirstIn1stServe', 'l_FirstIn1stServe', 
        'w_FirstWonFirstIn', 'l_FirstWonFirstIn', 'w_SecondWonSecondIn', 'l_SecondWonSecondIn',
        'w_cum_minutes', 'l_cum_minutes', 'w_ClutchFactor', 'l_ClutchFactor',
        'winner_glicko2', 'loser_glicko2'
    ]
    data = data.drop(columns=[c for c in raw_cols_to_drop if c in data.columns])

    return data
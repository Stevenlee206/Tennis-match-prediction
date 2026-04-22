import pandas as pd

def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    data = df.copy()
    missing_ratio = data.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    data = data.drop(columns=cols_to_drop)
    return data

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in ['winner_rank', 'loser_rank', 'winner_ht', 'loser_ht']:
        data[f'{col}_missing'] = data[col].isna().astype(int)
    for col in ['winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points']:
        data[col] = data.groupby('surface')[col].transform(
            lambda x: x.fillna(x.median())
        )
        data[col] = data[col].fillna(data[col].median())
    for col in ['winner_ht', 'loser_ht']:
        ioc_col = col.replace('_ht', '_ioc')
        data[col] = data.groupby(ioc_col)[col].transform(
            lambda x: x.fillna(x.median())
        )
        data[col] = data[col].fillna(data[col].median())  
    for col in ['winner_age', 'loser_age']:
        data[col] = data[col].fillna(data[col].median())
    data['surface'] = data['surface'].fillna('Unknown')

    data['winner_hand'] = data['winner_hand'].fillna('R')
    data['loser_hand']  = data['loser_hand'].fillna('R')

    stat_cols = [c for c in data.columns if c.startswith(('w_', 'l_'))]
    data = data.dropna(subset=stat_cols)
    return data

def remove_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    direct_leak = [
        'winner_id', 'loser_id',
        'winner_name', 'loser_name',
        'winner_ioc', 'loser_ioc',      
        'score',
        'minutes',                        
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon',
        'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon',
        'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    ]

    redundant_raw = [
        'winner_rank', 'loser_rank',           
        'winner_rank_points', 'loser_rank_points',  
        'winner_age', 'loser_age',             
        'winner_ht', 'loser_ht',               
        'winner_hand', 'loser_hand',           
        'winner_seed', 'loser_seed',           
        'winner_entry', 'loser_entry',         
    ]
    missing_flags_of_dropped = [
        'winner_ht_missing', 'loser_ht_missing',
    ]
    cols_to_drop = direct_leak + redundant_raw + missing_flags_of_dropped
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    data = df.drop(columns=existing_drops)
    return data

def remove_unused_data(data: pd.DataFrame) -> pd.DataFrame:
    COLS_TO_DROP = ['tourney_id', 'draw_size', 'tourney_date', 'match_num', 'rank_age_interaction',
        'winner_elo', 'loser_elo', 'winner_rank_missing', 'loser_rank_missing']
    return data.drop(columns = COLS_TO_DROP)
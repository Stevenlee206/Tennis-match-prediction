import pandas as pd

def drop_high_missing_columns(df: pd.DataFrame, train_idx: int, threshold: float = 0.1) -> pd.DataFrame:
    data = df.copy()
    # FIT: calculate missing ratio strictly on the train pool
    missing_ratio = data.iloc[:train_idx].isnull().mean()
    # TRANSFORM
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    # Protect columns that are essential for future feature generation
    protected_cols = ['minutes', 'w_bpSaved', 'w_bpFaced', 'l_bpSaved', 'l_bpFaced']
    cols_to_drop = [c for c in cols_to_drop if c not in protected_cols]
    
    data = data.drop(columns=cols_to_drop)
    return data

def fill_missing_values(df: pd.DataFrame, train_idx: int) -> pd.DataFrame:
    data = df.copy()
    train_slice = data.iloc[:train_idx] # The isolated train data
    
    for col in ['winner_rank', 'loser_rank', 'winner_ht', 'loser_ht']:
        data[f'{col}_missing'] = data[col].isna().astype(int)
        
    for col in ['winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points']:
        # FIT 
        surf_meds = train_slice.groupby('surface')[col].median()
        glob_med = train_slice[col].median()
        # TRANSFORM
        data[col] = data.apply(
            lambda row: surf_meds.get(row['surface'], glob_med) if pd.isna(row[col]) else row[col], axis=1
        )
        
    for col in ['winner_ht', 'loser_ht']:
        ioc_col = col.replace('_ht', '_ioc')
        # FIT
        ioc_meds = train_slice.groupby(ioc_col)[col].median()
        glob_med = train_slice[col].median()
        # TRANSFORM
        data[col] = data.apply(
            lambda row: ioc_meds.get(row.get(ioc_col, 'Unknown'), glob_med) if pd.isna(row[col]) else row[col], axis=1
        )  
        
    for col in ['winner_age', 'loser_age']:
        data[col] = data[col].fillna(train_slice[col].median())
        
    data['surface'] = data['surface'].fillna('Unknown')
    data['winner_hand'] = data['winner_hand'].fillna('R')
    data['loser_hand']  = data['loser_hand'].fillna('R')

    # (Note: dropna(subset=stat_cols) was removed here, we do it in preprocessing.py now)
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
        'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'w_2ndIn', 'l_2ndIn'
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
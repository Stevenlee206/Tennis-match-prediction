import pandas as pd
import numpy as np

def apply_historical_serve_metrics(df, train_idx: int):
    if not pd.api.types.is_datetime64_any_dtype(df['tourney_date']):
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

    df['w_2ndIn'] = df['w_svpt'] - df['w_1stIn'] - df['w_df']
    df['l_2ndIn'] = df['l_svpt'] - df['l_1stIn'] - df['l_df']
    df['match_idx'] = df.index
    winners = df[['match_idx', 'tourney_date', 'winner_id', 'w_ace', 'w_df', 'w_1stIn', 'w_svpt', 'w_1stWon', 'w_2ndIn', 'w_2ndWon']].copy()
    winners.columns = ['match_idx', 'tourney_date', 'player_id', 'ace', 'df', '1stIn', 'svpt', '1stWon', '2ndIn', '2ndWon']
    winners['is_winner'] = True
    losers = df[['match_idx', 'tourney_date', 'loser_id', 'l_ace', 'l_df', 'l_1stIn', 'l_svpt', 'l_1stWon', 'l_2ndIn', 'l_2ndWon']].copy()
    losers.columns = ['match_idx', 'tourney_date', 'player_id', 'ace', 'df', '1stIn', 'svpt', '1stWon', '2ndIn', '2ndWon']
    losers['is_winner'] = False
    timeline = pd.concat([winners, losers]).sort_values(['player_id', 'tourney_date'])
    timeline.index = timeline['tourney_date']
    roll_cols = ['ace', 'df', '1stIn', 'svpt', '1stWon', '2ndIn', '2ndWon']
    rolling_sums = timeline.groupby('player_id')[roll_cols].rolling('365D', min_periods=1).sum()
    for col in roll_cols:
        rolling_sums[col] = rolling_sums[col].values - timeline[col].values
        
    with np.errstate(divide='ignore', invalid='ignore'):
        timeline['AceVsDf'] = rolling_sums['ace'].values / rolling_sums['df'].values
        timeline['FirstIn1stServe'] = rolling_sums['1stIn'].values / rolling_sums['svpt'].values
        timeline['FirstWonFirstIn'] = rolling_sums['1stWon'].values / rolling_sums['1stIn'].values
        timeline['SecondWonSecondIn'] = rolling_sums['2ndWon'].values / rolling_sums['2ndIn'].values
    
    win_stats = timeline[timeline['is_winner']].set_index('match_idx')
    lose_stats = timeline[~timeline['is_winner']].set_index('match_idx')
    
    metrics = ['AceVsDf', 'FirstIn1stServe', 'FirstWonFirstIn', 'SecondWonSecondIn']
    for metric in metrics:
        df[f'w_{metric}'] = df['match_idx'].map(win_stats[metric])
        df[f'l_{metric}'] = df['match_idx'].map(lose_stats[metric])
        df[f'w_{metric}'] = df[f'w_{metric}'].replace([np.inf, -np.inf], np.nan)
        df[f'l_{metric}'] = df[f'l_{metric}'].replace([np.inf, -np.inf], np.nan)
        w_med = df[f'w_{metric}'].iloc[:train_idx].median()
        l_med = df[f'l_{metric}'].iloc[:train_idx].median()
        df[f'w_{metric}'] = df[f'w_{metric}'].fillna(w_med)
        df[f'l_{metric}'] = df[f'l_{metric}'].fillna(l_med)
    return df

def create_gao_dataset(df, train_idx: int):
    target_features = ['ht', 'age', 'AceVsDf', 'FirstIn1stServe', 'FirstWonFirstIn', 'SecondWonSecondIn']
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            train_median = df[col].iloc[:train_idx].median()
            df[col] = df[col].fillna(train_median)

    final_df = pd.DataFrame()
    final_df['year'] = df['tourney_date'].dt.year 
    targets = []
    
    np.random.seed(42)
    for index, row in df.iterrows():
        if np.random.rand() > 0.5:
            targets.append(1)
            for feat in target_features:
                w_col = f'winner_{feat}' if feat in ['ht', 'age'] else f'w_{feat}'
                l_col = f'loser_{feat}' if feat in ['ht', 'age'] else f'l_{feat}'                
                final_df.loc[index, f'{feat}_diff'] = row[w_col] - row[l_col]
        else:
            targets.append(0)
            for feat in target_features:
                w_col = f'winner_{feat}' if feat in ['ht', 'age'] else f'w_{feat}'
                l_col = f'loser_{feat}' if feat in ['ht', 'age'] else f'l_{feat}'
                final_df.loc[index, f'{feat}_diff'] = row[l_col] - row[w_col]
    final_df['target'] = targets
    return final_df
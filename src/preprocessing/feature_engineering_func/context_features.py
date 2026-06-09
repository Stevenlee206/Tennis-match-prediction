import pandas as pd
import numpy as np

def apply_fatigue_and_clutch_metrics(df, train_idx: int):
    median_mins = df['minutes'].iloc[:train_idx].median()
    df['minutes'] = df['minutes'].fillna(median_mins)
    df['match_idx'] = df.index

    winners = df[['match_idx', 'tourney_id', 'match_num', 'winner_id', 'minutes', 'w_bpSaved', 'w_bpFaced']].copy()
    winners.columns = ['match_idx', 'tourney_id', 'match_num', 'player_id', 'minutes', 'bpSaved', 'bpFaced']
    winners['is_winner'] = True
    
    losers = df[['match_idx', 'tourney_id', 'match_num', 'loser_id', 'minutes', 'l_bpSaved', 'l_bpFaced']].copy()
    losers.columns = ['match_idx', 'tourney_id', 'match_num', 'player_id', 'minutes', 'bpSaved', 'bpFaced']
    losers['is_winner'] = False
    
    timeline = pd.concat([winners, losers])
    timeline['bpSaved'] = timeline['bpSaved'].fillna(0)
    timeline['bpFaced'] = timeline['bpFaced'].fillna(0)
    date_map = df.set_index('match_idx')['tourney_date']
    timeline['tourney_date'] = timeline['match_idx'].map(date_map)
    timeline = timeline.sort_values(['player_id', 'tourney_date', 'match_num'])
    timeline['cum_minutes'] = timeline.groupby(['tourney_id', 'player_id'])['minutes'].cumsum() - timeline['minutes']
    timeline.index = timeline['tourney_date']
    roll_cols = ['bpSaved', 'bpFaced']
    rolling_sums = timeline.groupby('player_id')[roll_cols].rolling('365D', min_periods=1).sum()
    for col in roll_cols:
        rolling_sums[col] = rolling_sums[col].values - timeline[col].values
    with np.errstate(divide='ignore', invalid='ignore'):
        timeline['ClutchFactor'] = rolling_sums['bpSaved'].values / rolling_sums['bpFaced'].values
    win_stats = timeline[timeline['is_winner']].set_index('match_idx')
    lose_stats = timeline[~timeline['is_winner']].set_index('match_idx')
    
    for metric in ['cum_minutes', 'ClutchFactor']:
        df[f'w_{metric}'] = df['match_idx'].map(win_stats[metric])
        df[f'l_{metric}'] = df['match_idx'].map(lose_stats[metric])
        
        df[f'w_{metric}'] = df[f'w_{metric}'].replace([np.inf, -np.inf], np.nan)
        df[f'l_{metric}'] = df[f'l_{metric}'].replace([np.inf, -np.inf], np.nan)
        
        w_med = df[f'w_{metric}'].iloc[:train_idx].median()
        l_med = df[f'l_{metric}'].iloc[:train_idx].median()
        
        df[f'w_{metric}'] = df[f'w_{metric}'].fillna(w_med)
        df[f'l_{metric}'] = df[f'l_{metric}'].fillna(l_med)
        
    return df.drop(columns=['match_idx'])
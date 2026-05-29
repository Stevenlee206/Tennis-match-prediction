import pandas as pd
import numpy as np
from datetime import datetime

def apply_advanced_h2h(df: pd.DataFrame, alpha=0.01, c=2.0):
    """
    Calculates advanced Head-to-Head features with Time Decay and Laplace Smoothing.
    - alpha: decay factor per month (default 1%).
    - c: Laplace smoothing parameter (default 2).
    
    MUST be called on a dataframe sorted by `tourney_date`.
    """
    df = df.copy()
    
    # Ensure datetime format for calculation
    if not pd.api.types.is_datetime64_any_dtype(df['tourney_date']):
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    
    h2h_tracker = {}
    
    w_h2h_rates = []
    l_h2h_rates = []
    
    for idx, row in df.iterrows():
        w_id = row['winner_id']
        l_id = row['loser_id']
        current_date = row['tourney_date']
        
        # Symmetrical key
        p_min, p_max = min(w_id, l_id), max(w_id, l_id)
        
        if (p_min, p_max) not in h2h_tracker:
            h2h_tracker[(p_min, p_max)] = []
            
        history = h2h_tracker[(p_min, p_max)]
        
        # Calculate scores based on history (STRICTLY NO LEAKAGE: matches strictly before current_date)
        s_w = 0.0
        s_l = 0.0
        
        if pd.notna(current_date):
            for past_winner, past_date in history:
                if pd.isna(past_date):
                    continue
                
                # Delta in months (approx 30.44 days per month)
                delta_days = (current_date - past_date).days
                delta_months = max(0, delta_days / 30.44)
                
                weight = (1 - alpha) ** delta_months
                
                if past_winner == w_id:
                    s_w += weight
                else:
                    s_l += weight
                    
        # Apply Laplace smoothing formula
        rate_w = (s_w + c) / (s_w + s_l + 2 * c)
        rate_l = 1.0 - rate_w
        
        w_h2h_rates.append(rate_w)
        l_h2h_rates.append(rate_l)
        
        # AFTER calculating features, append the current match to history
        h2h_tracker[(p_min, p_max)].append((w_id, current_date))
        
    df['w_h2h_rate'] = w_h2h_rates
    df['l_h2h_rate'] = l_h2h_rates
    
    return df

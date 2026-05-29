import pandas as pd

def apply_matchup_topography(df):
    df = df.sort_values(['tourney_date']).copy()
    
    hand_tracker = {} 
    
    # Track only the differences
    hand_pct_diffs = []

    for idx, row in df.iterrows():
        w_id, l_id = row['winner_id'], row['loser_id']
        surface = row['surface'] if pd.notna(row['surface']) else 'Unknown'
        
        # ---> THE FIX: Strictly enforce L, R, or U <---
        w_hand = row['winner_hand'] if pd.notna(row['winner_hand']) and row['winner_hand'] in ['L', 'R'] else 'U'
        l_hand = row['loser_hand'] if pd.notna(row['loser_hand']) and row['loser_hand'] in ['L', 'R'] else 'U'

        # --- Handedness Difference ---
        if w_id not in hand_tracker: hand_tracker[w_id] = {'L': [0,0], 'R': [0,0], 'U': [0,0]}
        if l_id not in hand_tracker: hand_tracker[l_id] = {'L': [0,0], 'R': [0,0], 'U': [0,0]}

        w_stats = hand_tracker[w_id][l_hand]
        w_win_pct = w_stats[0] / w_stats[1] if w_stats[1] > 0 else 0.5
        
        l_stats = hand_tracker[l_id][w_hand]
        l_win_pct = l_stats[0] / l_stats[1] if l_stats[1] > 0 else 0.5

        # Winner's pct minus Loser's pct
        hand_pct_diffs.append(w_win_pct - l_win_pct)

        hand_tracker[w_id][l_hand][0] += 1 
        hand_tracker[w_id][l_hand][1] += 1 
        hand_tracker[l_id][w_hand][1] += 1 

    # Attach only the differences to the dataframe
    df['hand_win_pct_diff'] = hand_pct_diffs

    return df
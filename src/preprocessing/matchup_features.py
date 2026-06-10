import pandas as pd

def apply_matchup_topography(df : pd.DataFrame):
    df = df.sort_values(['tourney_date']).copy()
    
    h2h_tracker = {} 
    hand_tracker = {} 
    
    # Track only the differences
    h2h_diffs = []
    hand_pct_diffs = []

    for idx, row in df.iterrows():
        w_id, l_id = row['winner_id'], row['loser_id']
        surface = row['surface'] if pd.notna(row['surface']) else 'Unknown'
        
        # ---> THE FIX: Strictly enforce L, R, or U <---
        w_hand = row['winner_hand'] if pd.notna(row['winner_hand']) and row['winner_hand'] in ['L', 'R'] else 'U'
        l_hand = row['loser_hand'] if pd.notna(row['loser_hand']) and row['loser_hand'] in ['L', 'R'] else 'U'

        # --- H2H Difference ---
        p_min, p_max = min(w_id, l_id), max(w_id, l_id)
        if (p_min, p_max) not in h2h_tracker:
            h2h_tracker[(p_min, p_max)] = {p_min: 0, p_max: 0}

        current_w_wins = h2h_tracker[(p_min, p_max)][w_id]
        current_l_wins = h2h_tracker[(p_min, p_max)][l_id]
        
        # Winner's wins minus Loser's wins
        h2h_diffs.append(current_w_wins - current_l_wins)
        h2h_tracker[(p_min, p_max)][w_id] += 1

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
    df['h2h_advantage_diff'] = h2h_diffs
    df['hand_win_pct_diff'] = hand_pct_diffs

    return df
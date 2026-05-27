import pandas as pd
import numpy as np

def categorize_players(data: pd.DataFrame, num_players_per_category=3):
    """
    Categorizes players based on their mean Elo before and after 2024.
    Returns a dictionary of categories mapped to lists of player names.
    """
    # Ensure date column is datetime
    data['tourney_date'] = pd.to_datetime(data['tourney_date'], errors='coerce')
    
    pre_2024 = data[data['tourney_date'].dt.year.isin(range(2014, 2024 + 1))]
    post_2024 = data[data['tourney_date'].dt.year >= 2024]
    
    def get_player_elos(df):
        elos = {}
        for _, row in df.iterrows():
            if pd.notna(row.get('winner_name')) and pd.notna(row.get('winner_elo')):
                elos.setdefault(row['winner_name'], []).append(row['winner_elo'])
            if pd.notna(row.get('loser_name')) and pd.notna(row.get('loser_elo')):
                elos.setdefault(row['loser_name'], []).append(row['loser_elo'])
        
        # Calculate mean Elo for each player, filter out those with too few matches
        mean_elos = {p: np.mean(e) for p, e in elos.items() if len(e) >= 10}
        return mean_elos, elos
        
    pre_elos_mean, pre_elos_raw = get_player_elos(pre_2024)
    post_elos_mean, post_elos_raw = get_player_elos(post_2024)
    
    # Find intersection of players who played both before and after 2024
    common_players = set(pre_elos_mean.keys()).intersection(set(post_elos_mean.keys()))
    
    pre_elo_series = pd.Series({p: pre_elos_mean[p] for p in common_players})
    post_elo_series = pd.Series({p: post_elos_mean[p] for p in common_players})
    
    # Calculate quantiles based on the common pool
    pre_q3 = pre_elo_series.quantile(0.75)
    pre_q2 = pre_elo_series.quantile(0.5)
    pre_q1 = pre_elo_series.quantile(0.25)
    post_q3 = post_elo_series.quantile(0.75)
    post_q2 = post_elo_series.quantile(0.5)
    post_q1 = post_elo_series.quantile(0.25)
    
    print(f"Pre-2024 Elo Thresholds: Q1={pre_q1:.1f}, Q2={pre_q2:.1f}, Q3={pre_q3:.1f}")
    print(f"Post-2024 Elo Thresholds: Q1={post_q1:.1f}, Q2={post_q2:.1f}, Q3={post_q3:.1f}")
    
    categories = {
        'Good': [],
        'Bad': [],
        'Declining': [],
        'Rising': [],
        'Erratic': []
    }
    
    # 1. Good: Top 25% (Q3) in both
    for p in common_players:
        if pre_elo_series[p] >= pre_q3 and post_elo_series[p] >= post_q3:
            categories['Good'].append(p)
            
    # 2. Bad: Bottom 25% (Q1) in both
    for p in common_players:
        if pre_elo_series[p] <= pre_q1 and post_elo_series[p] <= post_q1:
            categories['Bad'].append(p)
            
    # 3. Declining: Top 50% (Q2) -> Bottom 25% (Q1)
    for p in common_players:
        if pre_elo_series[p] >= pre_q2 and post_elo_series[p] <= post_q1:
            categories['Declining'].append(p)
            
    # 4. Rising: Bottom 50% (Q2) -> Top 25% (Q3)
    for p in common_players:
        if pre_elo_series[p] <= pre_q2 and post_elo_series[p] >= post_q3:
            categories['Rising'].append(p)
            
    # 5. Erratic: Highest variance in Elo overall
    all_elos_mean, all_elos_raw = get_player_elos(data)
    variance_series = pd.Series({p: np.var(e) for p, e in all_elos_raw.items() if len(e) >= 30})
    erratic_candidates = variance_series.nlargest(20).index.tolist()
    # Ensure they actually played in the post_2024 period so we can test them
    for p in erratic_candidates:
        if p in post_elos_mean:
            categories['Erratic'].append(p)

    # Final selection: Pick top N by some metric or just random
    selected_players = {}
    
    # For declining, sort by how much they declined
    if categories['Declining']:
        categories['Declining'].sort(key=lambda p: pre_elo_series[p] - post_elo_series[p], reverse=True)
        
    # For rising, sort by how much they rose
    if categories['Rising']:
        categories['Rising'].sort(key=lambda p: post_elo_series[p] - pre_elo_series[p], reverse=True)
        
    # For good, sort by total matches to ensure we have data
    if categories['Good']:
        categories['Good'].sort(key=lambda p: len(pre_elos_raw[p]) + len(post_elos_raw[p]), reverse=True)

    # For bad, sort by total matches to ensure we have data
    if categories['Bad']:
        categories['Bad'].sort(key=lambda p: len(pre_elos_raw[p]) + len(post_elos_raw[p]), reverse=True)
        
    for cat, p_list in categories.items():
        selected_players[cat] = p_list[:num_players_per_category]
        
    # Flatten the list of selected target players
    all_target_players = []
    print("\n--- Selected Target Players ---")
    for cat, p_list in selected_players.items():
        print(f"{cat}: {p_list}")
        all_target_players.extend(p_list)
        
    return selected_players, all_target_players

if __name__ == '__main__':
    from data_splitter import BenchmarkDataPipeline
    
    pipeline = BenchmarkDataPipeline()
    data, _, _ = pipeline.run()
    
    cats, targets = categorize_players(data, num_players_per_category=2)

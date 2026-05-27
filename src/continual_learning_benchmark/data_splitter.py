import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import Preprocessing
from src.preprocessing.load_data import load_data
from src.preprocessing.target_encoding import create_target
from src.preprocessing.feature_engineering import build_elo_feature, build_glicko2_feature, build_recent_form, build_basic_features
from src.preprocessing.clean_data import drop_high_missing_columns, fill_missing_values, remove_leaky_columns, remove_unused_data
from src.config.data_config import CLEAN_THRESHOLD
from src.preprocessing.encoder import encode_categorical_features
from src.preprocessing.feature_selection import feature_selection
from src.preprocessing.context_features import apply_fatigue_and_clutch_metrics
from src.preprocessing.gao_features import apply_historical_serve_metrics
from src.preprocessing.matchup_features import apply_matchup_topography

class BenchmarkDataPipeline(Preprocessing):
    """
    Custom pipeline for the Continual Learning Benchmark.
    Ensures that data from 2014 to 2026 is loaded, and the 
    leakage-prevention split is strictly set to 90% of the pre-2025 data.
    """
    def __init__(self):
        super().__init__()
        self.years_to_load = list(range(2014, 2027)) # 2014 to 2026
        
    def _load(self):
        dfs = []
        for year in self.years_to_load:
            file_name = f"atp_matches_{year}.csv"
            # Some future years might not exist in raw_data if not downloaded, handle gracefully
            try:
                df_year = load_data(file_name)
                dfs.append(df_year)
            except Exception as e:
                print(f"Skipping {file_name} as it might not exist: {e}")

        self.data = pd.concat(dfs, axis=0, ignore_index=True)
        return self.data
        
    def run(self):
        data = self._load()
        
        stat_cols = [c for c in data.columns if c.startswith(('w_', 'l_'))]
        data = data.dropna(subset=stat_cols).copy()

        if 'tourney_date' in data.columns:
            data['tourney_date'] = pd.to_datetime(data['tourney_date'], format='%Y%m%d', errors='coerce')
        data = data.sort_values('tourney_date').reset_index(drop=True)
        
        # Calculate the Train Split Threshold dynamically: 90% of pre-2025 data
        pre_2025_mask = data['tourney_date'].dt.year < 2025
        pre_2025_count = pre_2025_mask.sum()
        train_split_idx = int(pre_2025_count * 0.90)
        
        print(f"Total rows: {len(data)}, Pre-2025 rows: {pre_2025_count}, Base Train Split Index: {train_split_idx}")

        # Pass the train_split_idx to leaky functions (Fit on Train, Apply to All)
        data = drop_high_missing_columns(data, train_split_idx, threshold=CLEAN_THRESHOLD)
        data = fill_missing_values(data, train_split_idx)
        
        data = build_basic_features(data)
        data = build_elo_feature(data)
        data = build_glicko2_feature(data)
        data = build_recent_form(data)
        
        print("Calculating Historical Serve Metrics...")
        data = apply_historical_serve_metrics(data, train_split_idx)
        data = apply_fatigue_and_clutch_metrics(data, train_split_idx)
        data = apply_matchup_topography(data)
        
        data = create_target(data, augment=True)
        
        # Since augment=True perfectly interleaves the duplicate matches,
        # the dataset size is doubled. We must double our indices to keep them aligned.
        train_split_idx *= 2
        pre_2025_count *= 2
        
        # Preserve columns needed for Player Categorization before they get dropped
        dates = data['tourney_date'].copy()
        w_elos = data['winner_elo'].copy() if 'winner_elo' in data.columns else None
        l_elos = data['loser_elo'].copy() if 'loser_elo' in data.columns else None
        
        data = remove_leaky_columns(data)
        data = remove_unused_data(data)
        data = encode_categorical_features(data)
        
        names = data[['winner_name', 'loser_name']].copy() if 'winner_name' in data.columns else None
        data = feature_selection(data, train_split_idx, k=40) 
        
        # Restore them
        if names is not None:
            data[['winner_name', 'loser_name']] = names
        data['tourney_date'] = dates
        if w_elos is not None:
            data['winner_elo'] = w_elos
            data['loser_elo'] = l_elos
            
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            data[nan_cols] = data[nan_cols].fillna(0)
            
        return data, train_split_idx, pre_2025_count

def get_benchmark_splits(data: pd.DataFrame, train_split_idx: int, pre_2025_count: int, target_players=None):
    """
    Slices the processed data into D_Base, D_Stream, D_Test, and D_Holdout.
    Dynamically expands D_Test (up to 30% of 2026) to ensure target players have matches.
    
    Args:
        data: Processed DataFrame
        train_split_idx: Index separating D_Base and the 10% pre-2025 val set.
        pre_2025_count: Total number of matches before 2025.
        target_players: List of player names we need to ensure are in D_Test.
    """
    
    # D_Base: 90% of pre-2025
    D_Base = data.iloc[:train_split_idx].copy()
    
    # 2026 data boundary
    # We define "2026 data" as anything in year 2026. 
    
    is_2026 = data['tourney_date'].dt.year >= 2026
    if not is_2026.any():
        # Fallback if 2026 data isn't loaded: use 2025 as the proxy for 2026
        is_2026 = data['tourney_date'].dt.year >= 2025

    idx_2026_start = data[is_2026].index[0]
    total_2026_matches = len(data) - idx_2026_start
    
    # Determine how much of 2026 goes into D_Test. Default is 10%. Max is 30%.
    # The first 10% goes to D_Test. But D_Stream gets "Toàn bộ đầu năm 2026". 
    # Let's clarify: D_Test is 10% of 2026. If it's the *next* 10%, that implies D_Stream takes some portion of early 2026.
    # Let's say D_Stream takes the first 20% of 2026, and D_Test takes the next 10% to 30%.
    # Actually, simpler: D_Test is the NEXT 10-30% of 2026. 
    # Let's just say D_Stream goes up to the start of D_Test.
    
    # To keep it robust, let's say D_Test starts at the beginning of 2026. 
    # Wait, user said: "Toàn bộ đầu năm 2026", "Dùng 10% tiếp theo của năm 2026 để test".
    # This means D_Stream takes a chunk of 2026, then D_Test takes the next chunk.
    # Let's allocate the first 10% of 2026 to D_Stream.
    idx_stream_end = idx_2026_start + int(0.10 * total_2026_matches)
    
    # Now find D_Test boundaries
    test_pct = 0.10
    found_all = False
    
    while test_pct <= 0.5 and not found_all:
        idx_test_end = idx_stream_end + int(test_pct * total_2026_matches)
        D_Test_candidate = data.iloc[idx_stream_end:idx_test_end]
        
        if target_players:
            found_all = True
            for p in target_players:
                p_matches = D_Test_candidate[(D_Test_candidate['winner_name'] == p) | (D_Test_candidate['loser_name'] == p)]
                if len(p_matches) == 0:
                    found_all = False
                    break
            
            if not found_all:
                test_pct += 0.05 # Increment by 5% up to 30%
        else:
            found_all = True
            
    idx_test_end = min(idx_stream_end + int(test_pct * total_2026_matches), len(data))
    
    D_Stream = data.iloc[train_split_idx:idx_stream_end].copy()
    D_Test = data.iloc[idx_stream_end:idx_test_end].copy()
    D_Holdout = data.iloc[idx_test_end:].copy()
    
    # We must ensure we don't have augmented rows in D_Test/D_Holdout affecting evaluation
    if 'is_augmented' in D_Test.columns:
        D_Test = D_Test[D_Test['is_augmented'] == 0]
        D_Holdout = D_Holdout[D_Holdout['is_augmented'] == 0]
        
    return D_Base, D_Stream, D_Test, D_Holdout

if __name__ == '__main__':
    pipeline = BenchmarkDataPipeline()
    data, train_split_idx, pre_2025_count = pipeline.run()
    base, stream, test, holdout = get_benchmark_splits(data, train_split_idx, pre_2025_count)
    print(f"Base: {len(base)}, Stream: {len(stream)}, Test: {len(test)}, Holdout: {len(holdout)}")

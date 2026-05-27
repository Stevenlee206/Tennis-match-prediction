import pandas as pd
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.load_data import load_data
from src.preprocessing.target_encoding import create_target
from src.preprocessing.feature_engineering import build_elo_feature, build_glicko2_feature, build_recent_form, build_basic_features
from src.preprocessing.clean_data import drop_high_missing_columns, fill_missing_values, remove_leaky_columns, remove_unused_data
from src.config.data_config import YEARS, CLEAN_THRESHOLD
from src.preprocessing.encoder import encode_categorical_features
from src.preprocessing.feature_selection import feature_selection
from src.preprocessing.context_features import apply_fatigue_and_clutch_metrics
# ---> IMPORT ONLY THE SERVE METRICS FUNCTION <---
from src.preprocessing.gao_features import apply_historical_serve_metrics
from src.preprocessing.matchup_features import apply_matchup_topography
class Preprocessing:
    def __init__(self):
        self.data = None

    def _load(self):
        dfs = []
        for year in YEARS:
            file_name = f"atp_matches_{year}.csv"
            df_year = load_data(file_name)
            dfs.append(df_year)

        self.data = pd.concat(dfs, axis=0, ignore_index=True)
        return self.data
    
    def run(self, train_ratio=0.8):
        data = self._load()
        
        # 1. IMPORTANT: Drop missing stats early so the dataset length stabilizes
        stat_cols = [c for c in data.columns if c.startswith(('w_', 'l_'))]
        data = data.dropna(subset=stat_cols).copy()

        # 2. Sort chronologically to mirror main.py's `shuffle=False`
        if 'tourney_date' in data.columns:
            data['tourney_date'] = pd.to_datetime(data['tourney_date'], format='%Y%m%d', errors='coerce')
        data = data.sort_values('tourney_date').reset_index(drop=True)
        
        # 3. Calculate the Train Split Threshold dynamically
        train_split_idx = int(len(data) * train_ratio)
        
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
        
        data = create_target(data, augment = True)
        
        data = remove_leaky_columns(data)
        data = remove_unused_data(data)
        data = encode_categorical_features(data)
        
        current_train_idx = int(len(data) * train_ratio)
        
        # Pass the NEW index to correlation selector
        names = data[['winner_name', 'loser_name']].copy() if 'winner_name' in data.columns else None
        data = feature_selection(data, current_train_idx, k=40) 
        if names is not None:
            data[['winner_name', 'loser_name']] = names
        
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            print(f"\nWARNING: Sneaky NaNs detected in columns: {nan_cols}")
            print("Applying safety net imputation (filling with 0) so Optuna doesn't crash...")
            
            # Since most of your final features are target-encoded differences (e.g., elo_diff, rank_diff),
            # filling a missing difference with 0 is statistically safe because 0 implies "no advantage to either player".
            data[nan_cols] = data[nan_cols].fillna(0)
        return data

if __name__ == '__main__':
    p = Preprocessing()
    final_data = p.run()
    print(f"Preprocessing complete. Final shape: {final_data.shape}")
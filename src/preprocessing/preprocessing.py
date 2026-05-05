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
    
    def run(self):
        data = self._load()
        
        # 1. Clean and build basic/advanced features
        data = drop_high_missing_columns(data, threshold=CLEAN_THRESHOLD)
        data = fill_missing_values(data)
        data = build_basic_features(data)
        data = build_elo_feature(data)
        data = build_glicko2_feature(data)
        data = build_recent_form(data)
        
        # 2. Add Gao's Historical Serve Strengths BEFORE target encoding
        print("Calculating Historical Serve Metrics...")
        data = apply_historical_serve_metrics(data)
        data = apply_fatigue_and_clutch_metrics(data)
        # 3. Create target (This now calculates differences for ELO, Form, AND Serve Metrics)
        data = create_target(data)
        
        # 4. Final Cleanup & Selection
        data = remove_leaky_columns(data)
        data = remove_unused_data(data)
        data = encode_categorical_features(data)
        
        # Select the top K features (you can increase k=10 to k=15 in feature_selection.py to keep more)
        data = feature_selection(data, k=40) 
        
        return data

if __name__ == '__main__':
    p = Preprocessing()
    final_data = p.run()
    print(f"Preprocessing complete. Final shape: {final_data.shape}")
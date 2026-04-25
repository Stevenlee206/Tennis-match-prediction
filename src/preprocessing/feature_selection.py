import pandas as pd
import numpy as np

def feature_selection(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Performs feature selection by keeping the top 'k' features 
    most highly correlated (absolute value) with the target.
    """
    data = df.copy()

    # 1. Safety Net: Drop any remaining object/string columns
    data = data.select_dtypes(exclude=['object', 'category'])
    
    # 2. Remove zero-variance features (columns where all values are identical)
    cols_to_check = [c for c in data.columns if c != 'target']
    variances = data[cols_to_check].var()
    
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        data = data.drop(columns=zero_var_cols)
        
    # 3. Correlation-based Selection (Absolute values)
    # Calculate Pearson correlation with the target
    correlations = data.corr()['target'].drop('target')
    
    # Drop features that returned NaN (usually due to zero variance 
    # appearing after earlier splits, like tourney_level_O)
    correlations = correlations.dropna()
    
    # Sort by absolute correlation to find the strongest signals (positive OR negative)
    top_k_features = correlations.abs().sort_values(ascending=False).head(k).index.tolist()
    
    print(f"\n[Feature Selection] Keeping Top {k} features based on absolute correlation:")
    for feat in top_k_features:
        # We print the original correlation (with sign) to see direction
        print(f"  - {feat}: {correlations[feat]:.4f}")
        
    # Keep only the selected features + the target + the year (needed for walk-forward)
    cols_to_keep = top_k_features + ['target']
    
    # We must preserve the 'year' column if it exists so Walk-Forward validation doesn't crash
    if 'year' in data.columns and 'year' not in cols_to_keep:
        cols_to_keep.append('year')
        
    data = data[cols_to_keep]
        
    return data
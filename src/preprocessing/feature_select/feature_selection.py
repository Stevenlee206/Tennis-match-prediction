import pandas as pd
import numpy as np

def feature_selection(df: pd.DataFrame, train_idx: int, k: int = 10) -> pd.DataFrame:
    data = df.copy()
    data = data.select_dtypes(exclude=['object', 'category'])
    
    train_slice = data.iloc[:train_idx]
    
    cols_to_check = [c for c in train_slice.columns if c != 'target']
    variances = train_slice[cols_to_check].var()
    
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        data = data.drop(columns=zero_var_cols)
        train_slice = train_slice.drop(columns=zero_var_cols)
        
    correlations = train_slice.corr()['target'].drop('target').dropna()
    top_k_features = correlations.abs().sort_values(ascending=False).head(k).index.tolist()
    
    cols_to_keep = top_k_features + ['target']
    
    if 'year' in data.columns and 'year' not in cols_to_keep:
        cols_to_keep.append('year')
        
    if 'is_augmented' in data.columns and 'is_augmented' not in cols_to_keep:
        cols_to_keep.append('is_augmented')
        
    return data[cols_to_keep]
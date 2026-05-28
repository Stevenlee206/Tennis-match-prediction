import pandas as pd

def feature_selection(df: pd.DataFrame, train_idx: int, k: int = 10) -> pd.DataFrame:
    """
    Selects the top 'k' numeric features with the highest absolute Pearson correlation to the target variable.
    To prevent data leakage, zero-variance checks and correlation calculations are fitted strictly on the training subset
    before filtering the entire dataset, while preserving essential metadata columns (target, year, is_augmented).
    """
    data = df.copy()
    data = data.select_dtypes(exclude=['object', 'category'])
    
    # Isolate Training Pool
    train_slice = data.iloc[:train_idx]
    
    # Fit zero-variance check on TRAIN
    cols_to_check = [c for c in train_slice.columns if c != 'target']
    variances = train_slice[cols_to_check].var()
    
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        data = data.drop(columns=zero_var_cols)
        train_slice = train_slice.drop(columns=zero_var_cols)
        
    # 3. Fit Correlation on TRAIN
    correlations = train_slice.corr()['target'].drop('target').dropna()
    top_k_features = correlations.abs().sort_values(ascending=False).head(k).index.tolist()
    
    cols_to_keep = top_k_features + ['target']
    
    if 'year' in data.columns and 'year' not in cols_to_keep:
        cols_to_keep.append('year')
        
    if 'is_augmented' in data.columns and 'is_augmented' not in cols_to_keep:
        cols_to_keep.append('is_augmented')
        
    return data[cols_to_keep]
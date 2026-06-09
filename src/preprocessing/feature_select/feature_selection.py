import pandas as pd

def feature_selection(df: pd.DataFrame, train_idx: int, k: int = 10) -> pd.DataFrame:
    data = df.copy()
    data = data.select_dtypes(exclude=['object', 'category'])
    train_slice = data.iloc[:train_idx]
    cols_to_check = [c for c in train_slice.columns if c != 'target']
    variances = train_slice[cols_to_check].var()
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        data = data.drop(columns=zero_var_cols)
    return data
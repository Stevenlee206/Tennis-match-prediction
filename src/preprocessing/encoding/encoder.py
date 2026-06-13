import pandas as pd

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features into numeric formats suitable for ML models.
    """
    data = df.copy()
    cols_to_drop = ['tourney_name', 'tourney_date'] 
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    categorical_cols = ['surface', 'tourney_level', 'round']
    existing_cats = [c for c in categorical_cols if c in data.columns]
    
    if existing_cats:
        data = pd.get_dummies(data, columns=existing_cats, drop_first=False)
        for col in data.columns:
            if data[col].dtype == bool:
                data[col] = data[col].astype(int)

    return data
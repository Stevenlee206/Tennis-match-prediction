import pandas as pd

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features into numeric formats suitable for ML models.
    """
    data = df.copy()

    # High-cardinality strings to drop (too many unique values for One-Hot Encoding)
    # tourney_date is handled in feature_engineering_func; tourney_name causes bloat.
    cols_to_drop = ['tourney_name', 'tourney_date'] 
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # One-Hot Encode low-cardinality categorical variables
    categorical_cols = ['surface', 'tourney_level', 'round']
    existing_cats = [c for c in categorical_cols if c in data.columns]
    
    if existing_cats:
        # Convert to one-hot variables
        data = pd.get_dummies(data, columns=existing_cats, drop_first=False)
        
        # Pandas get_dummies returns boolean (True/False). 
        # SVMs require strictly numeric data, so we cast booleans to integers (1/0).
        for col in data.columns:
            if data[col].dtype == bool:
                data[col] = data[col].astype(int)

    return data
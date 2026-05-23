import pandas as pd

def encode_features(data):
    columns_to_encode = ['surface', 'tourney_level']
    cols_after = [col for col in columns_to_encode if col in data.columns]
    
    if cols_after:
        data = pd.get_dummies(data, columns=cols_after, dtype=int)       
    return data
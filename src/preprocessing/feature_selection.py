import pandas as pd
from src.config.data_config import IMPORTANCE_COLUMNS

def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    safe_cols = [col for col in IMPORTANCE_COLUMNS if col in data.columns]
    return data[safe_cols]
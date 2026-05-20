import pandas as pd
from src.config.data_config import IMPORTANCE_COLUMNS

def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    cols = IMPORTANCE_COLUMNS.copy()
    # Keep year for time-based split; not necessarily a model feature.
    if 'year' in data.columns and 'year' not in cols:
        cols = ['year'] + cols
    return data[cols + ['target']]

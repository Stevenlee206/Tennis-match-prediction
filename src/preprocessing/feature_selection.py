import pandas as pd
from config.data_config import IMPORTANCE_COLUMNS

def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    return data[IMPORTANCE_COLUMNS + ['target']]

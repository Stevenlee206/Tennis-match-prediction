import pandas as pd
from src.preprocessing.load_data import load_data
from src.preprocessing.target_encoding import create_target
# from src.preprocessing.feature_selection import feature_selection
from src.preprocessing.feature_engineering import build_elo_feature, build_recent_form, build_basic_features
from src.preprocessing.clean_data import drop_high_missing_columns, fill_missing_values, remove_leaky_columns, remove_unused_data
from src.config.data_config import YEARS, CLEAN_THRESHOLD

class Preprocessing:
	def __init__(self):
		self.data = None

	def _load(self):
		dfs = []
		for year in YEARS:
			file_name = f"atp_matches_{year}.csv"
			df_year = load_data(file_name)
			dfs.append(df_year)

		self.data = pd.concat(dfs, axis=0, ignore_index=True)
		return self.data

	def _process_categorical(self, data):
		if 'tourney_name' in data.columns:
			data = data.drop(columns=['tourney_name'])

		cat_cols = ['surface', 'tourney_level', 'round']
		data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

		return data.astype('float32')

	def run(self):
		data = self._load()
		data = drop_high_missing_columns(data, threshold=CLEAN_THRESHOLD)
		data = fill_missing_values(data)
		data = build_basic_features(data)
		data = build_elo_feature(data)
		data = build_recent_form(data)
		data = create_target(data)
		data = remove_leaky_columns(data)
		data = remove_unused_data(data)
		# data = feature_selection(data)

		data = self._process_categorical(data)
		return data
preprocessor = Preprocessing()
df = preprocessor.run()

print(df.info()) # Kiểm tra kiểu dữ liệu
print(df.isnull().sum().sum()) # Kiểm tra còn giá trị thiếu không
print(df['target'].value_counts()) # Kiểm tra độ cân bằng nhãn
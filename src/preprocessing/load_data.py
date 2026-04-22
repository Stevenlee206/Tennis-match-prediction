import os
import pandas as pd

def load_data(filename='atp_matches_futures_2024.csv'):
	data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw_data')
	file_path = os.path.join(data_dir, filename)
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
	df = pd.read_csv(file_path)
	return df

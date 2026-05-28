import os
import pandas as pd
# Load Data
def load_data(filename='atp_matches_2024.csv')->pd.DataFrame:
	project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
	data_dir = os.path.join(project_root, 'data', 'raw_data')
	file_path = os.path.join(data_dir, filename)
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
	df = pd.read_csv(file_path)
	return df

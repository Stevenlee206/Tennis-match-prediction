import pandas as pd
from models.logistic_reg.log_reg_model import build_model
from models.logistic_reg.trainer import time_series_cv

def run_logistic_regression():
    print("Loading data")
    df = pd.read_csv('data/processed/final_ml_matrix.csv')
    
    # Seperating X and y
    X = df.drop(columns=['target'])
    y = df['target']

    # Initialize the model and grid from log_reg_model.py
    tennis_model = build_model(C=1.0, solver='lbfgs', max_iter=1000)

    # Start walk-foward time series
    report = time_series_cv(model=tennis_model, X=X, y=y)

    # Print terminal report
    print("\nSummary report")
    print(report.to_string(index=False))

if __name__ == "__main__":
    run_logistic_regression()
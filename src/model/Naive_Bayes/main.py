import pandas as pd
from src.model.Logistic_Regression.trainer import time_series_cv
from src.model.Naive_Bayes.naive_bayes_model import build_naive_bayes_model
def run_naive_bayes():
    print("Loading data")
    df = pd.read_csv('data/processed/final_ml_matrix.csv')
    X = df.drop(columns=['target'])
    y = df['target']
 
    # Initialize the model from naive_bayes_model.py
    nb_model = build_naive_bayes_model(var_smoothing=1e-9)
 
    # Start walk-foward time series
    report = time_series_cv(model=nb_model, X=X, y=y, results_dir="results/Naive_Bayes")
 
    # Print terminal report
    print("\nSummary report")
    print(report.to_string(index=False))

if __name__ == "__main__":
    run_naive_bayes()
import pandas as pd
from src.model.Logistic_Regression.log_reg_model import build_model
from src.model.Logistic_Regression.trainer import train_and_evaluate

def run_logistic_regression():
    print("Loading data")
    df = pd.read_csv('data/processed/final_ml_matrix.csv')
    

    # Seperating X and y
    X = df.drop(columns=['target'])
    y = df['target']

    # Initialize the model and grid from log_reg_model.py
    model, param_grid = build_model()

    # Execute
    final_model = train_and_evaluate(X, y, model, param_grid)

if __name__ == "__main__":
    run_logistic_regression()
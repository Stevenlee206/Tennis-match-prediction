from sklearn.linear_model import LogisticRegression
def build_model():
    model = LogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    
    return model, param_grid

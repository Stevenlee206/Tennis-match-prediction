from sklearn.linear_model import LogisticRegression
def build_model(**kwargs):
    param_grid = {
        'C': kwargs.get('C', 1.0),
        'solver': kwargs.get('solver', 'lbfgs'),
        'max_iter': kwargs.get('max_iter', 1000),
        'random_state': kwargs.get('random_state', 42)
    }
    
    return LogisticRegression(**param_grid)

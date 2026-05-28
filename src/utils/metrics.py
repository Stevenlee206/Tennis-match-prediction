from sklearn.metrics import accuracy_score


def accuracy_from_predictions(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

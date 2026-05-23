from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(X, y, model, param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Running GridSearchCV to find optimal parameters...")
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    print(f"Best Parameters Found: {grid_search.best_params_}")

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))
    
    return best_model
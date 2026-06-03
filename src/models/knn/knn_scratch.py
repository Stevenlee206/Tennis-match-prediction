import numpy as np
from collections import Counter

class KNNFromScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train, self.y_train = None, None

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

    def predict(self, X_test_matrix):
        X_test_matrix = np.array(X_test_matrix)
        return np.array([self._predict_single(x) for x in X_test_matrix])

    def _predict_single(self, x_test):
        distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_labels).most_common(1)[0][0]


# KIỂM THỬ ĐỘC LẬP
if __name__ == "__main__":
    print("--- Test KNNFromScratch ---")
    X_train_mock = np.array([[1, 2], [2, 3], [10, 11]])
    y_train_mock = np.array([0, 0, 1])
    X_test_mock = np.array([[1.5, 2.5], [9, 10]])
    
    knn = KNNFromScratch(k=1)
    knn.fit(X_train_mock, y_train_mock)
    preds = knn.predict(X_test_mock)
    print(f"Predictions (Expected [0, 1]): {preds}")

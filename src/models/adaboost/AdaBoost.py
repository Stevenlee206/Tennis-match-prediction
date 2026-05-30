import numpy as np
class Ada_Boost:
    def __init__(self,n_estimators: int = 50):
        self.n_estimators=n_estimators
        self.clfs: list[Decision_Stump] = []
        self.alphas: list[float] = []
    def fit(self, X: np.ndarray, y: np.ndarray, is_discrete_list: np.ndarray):
        n_samples, _ = X.shape
        w = np.full(n_samples, (1 / n_samples))
        for _ in range(self.n_estimators):
            clf = Decision_Stump(is_discrete_list)
            clf.fit(X, y)
            predictions = clf.predict(X)
            misclassified = (predictions != y)
            error = np.sum(w[misclassified])
            error = max(1e-9, min(1 - 1e-9, error))
            alpha = 0.5 * np.log((1.0 - error) / error)
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)
            self.alphas.append(alpha)
    def predict(self, X: np.ndarray):
        clf_preds = np.array([alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.clfs)])
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)
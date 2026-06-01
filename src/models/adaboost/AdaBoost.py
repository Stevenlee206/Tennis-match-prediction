import numpy as np
from src.models.adaboost.Decision_Stump import Decision_Stump  # Đã sửa lại đúng tên class

class AdaBoost:
    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.clfs: list[Decision_Stump] = []
        self.alphas: list[float] = []
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray, is_discrete_list: np.ndarray):
        n_samples = X.shape[0]

        # Lưu các class ban đầu để map ngược. Ép y về {-1, 1}
        self.classes = np.unique(y)
        if len(self.classes) == 2:
            y_mapped = np.where(y == self.classes[1], 1, -1)
        else:
            y_mapped = y.copy()

        # Khởi tạo trọng số
        w = np.full(n_samples, (1.0 / n_samples))

        for _ in range(self.n_estimators):
            clf = Decision_Stump(is_discrete_list)
            # Chuyền trọng số w vào DecisionStump
            clf.fit(X, y_mapped, w)

            predictions = clf.predict(X)
            misclassified = (predictions != y_mapped)

            error = np.sum(w[misclassified])
            error = max(1e-10, min(1.0 - 1e-10, error))  # Chống lỗi chia 0

            # Tính alpha (độ quan trọng của cây)
            alpha = 0.5 * np.log((1.0 - error) / error)

            # Cập nhật và chuẩn hóa trọng số
            w *= np.exp(-alpha * y_mapped * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)
            self.alphas.append(alpha)

    def predict(self, X: np.ndarray):
        clf_preds = np.array([alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.clfs)])
        y_pred = np.sign(np.sum(clf_preds, axis=0))

        # Trả về nhãn nguyên gốc ban đầu
        if self.classes is not None and len(self.classes) == 2:
            return np.where(y_pred == 1, self.classes[1], self.classes[0])

        return y_pred
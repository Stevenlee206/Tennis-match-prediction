import numpy as np
from src.models.adaboost.Decision_Node import Decision_Node
from src.models.adaboost.Decision_Tree_Metrics import gain
from src.models.adaboost.Get_most_common_weight import weighted_most_common

class Decision_Stump:
    def __init__(self, is_discrete_list: np.ndarray):
        self.root = None
        self.is_discrete_list = is_discrete_list

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        n_features = X.shape[1]
        best_feat_idx = -1
        best_gain = -1.0
        best_threshold = None

        for i in range(n_features):
            # Truyền w vào hàm gain
            current_gain, threshold = gain(y, X[:, i], self.is_discrete_list[i], w)
            if current_gain > best_gain:
                best_gain = current_gain
                best_feat_idx = i
                best_threshold = threshold

        self.root = Decision_Node(
            feature_idx=best_feat_idx,
            threshold=best_threshold,
            is_discrete=self.is_discrete_list[best_feat_idx]
        )

        feature_col = X[:, best_feat_idx]

        if self.root.is_discrete:
            for val in np.unique(feature_col):
                mask = (feature_col == val)
                # Dùng weighted_most_common thay vì count thông thường
                leaf_label = weighted_most_common(y[mask], w[mask])
                self.root.children[val] = Decision_Node(label=leaf_label)
        else:
            left_mask = feature_col < best_threshold
            right_mask = feature_col >= best_threshold
            self.root.children['left'] = Decision_Node(label=weighted_most_common(y[left_mask], w[left_mask]))
            self.root.children['right'] = Decision_Node(label=weighted_most_common(y[right_mask], w[right_mask]))

    def predict(self, X: np.ndarray):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x: np.ndarray, node: Decision_Node):
        if node.is_leaf():
            return node.label

        val = x[node.feature_idx]
        if node.is_discrete:
            return self._traverse(x, node.children.get(val, list(node.children.values())[0]))
        else:
            branch = 'left' if val < node.threshold else 'right'
            return self._traverse(x, node.children[branch])
import numpy as np
class DecisionStump:
    def __init__(self,is_discrete:np.array):
        self.root=None
        self.is_discrete=is_discrete
    def fit(self,X : np.array,y:np.array):
        n_features = X.shape[1]
        best_feat_idx = -1
        best_gain = -1.0
        best_threshold = None
        for i in range(n_features):
            current_gain, threshold = gain(y, X[:, i], self.is_discrete[i])
            if current_gain > best_gain :
                best_gain = current_gain
                best_feat_idx = i
                best_threshold = threshold
        self.root = Decision_Node(
            feature_idx=best_feat_idx,
            threshold=best_threshold,
            is_discrete=self.is_discrete_list[best_feat_idx]
        )
        feature_col=X[:,best_feat_idx]
        if self.root.is_discrete :
            for val in np.unique(feature_col):
                leaf_label = self._get_most_common(y[feature_col == val])
                self.root.children[val] = Decision_Node(value=leaf_label)
        else :
            left_y = y[feature_col < best_threshold]
            right_y = y[feature_col >= best_threshold]
            self.root.children['left'] = Decision_Node(value=self._get_most_common(left_y))
            self.root.children['right'] = Decision_Node(value=self._get_most_common(right_y))
    def _get_most_common(self,y : np.array):
        if len(y) == 0: return 0
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def predict(self, X: np.ndarray):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x: np.ndarray, node: Decision_Node):
        if node.is_leaf():
            return node.value
        val = x[node.feature_idx]
        if node.is_discrete:
            return self._traverse(x, node.children.get(val, list(node.children.values())[0]))
        else:
            branch = 'left' if val < node.threshold else 'right'
            return self._traverse(x, node.children[branch])

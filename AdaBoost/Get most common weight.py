import numpy as np
def weighted_most_common(y: np.ndarray, weights: np.ndarray):
    unique_labels = np.unique(y)
    best_label = None
    max_weight = -1
    for label in unique_labels:
        label_weight = np.sum(weights[y == label])
        if label_weight > max_weight:
            max_weight = label_weight
            best_label = label
    return best_label
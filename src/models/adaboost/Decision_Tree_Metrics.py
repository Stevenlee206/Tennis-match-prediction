import numpy as np


def entropy(labels: np.ndarray, w: np.ndarray) -> float:
    if len(labels) == 0 or np.sum(w) == 0:
        return 0.0

    w_total = np.sum(w)
    classes = np.unique(labels)
    ent = 0.0

    for c in classes:
        p_c = np.sum(w[labels == c]) / w_total
        if p_c > 0:
            ent -= p_c * np.log2(p_c)

    return ent


def gain(labels: np.ndarray, values: np.ndarray, is_discrete: bool, w: np.ndarray) -> tuple:
    base_entropy = entropy(labels, w)
    w_total = np.sum(w)

    if w_total == 0:
        return 0.0, None

    if is_discrete:
        unique_values = np.unique(values)
        weighted_ent = 0.0
        for val in unique_values:
            mask = (values == val)
            w_subset = np.sum(w[mask])
            if w_subset > 0:
                weighted_ent += (w_subset / w_total) * entropy(labels[mask], w[mask])
        return base_entropy - weighted_ent, None

    else:
        unique_values = np.unique(values)
        best_gain = -1.0
        best_threshold = None

        for val in unique_values:
            left_mask = values < val
            right_mask = values >= val

            w_left = np.sum(w[left_mask])
            w_right = np.sum(w[right_mask])

            ent_left = entropy(labels[left_mask], w[left_mask]) if w_left > 0 else 0
            ent_right = entropy(labels[right_mask], w[right_mask]) if w_right > 0 else 0

            weighted_ent = (w_left / w_total) * ent_left + (w_right / w_total) * ent_right
            current_gain = base_entropy - weighted_ent

            if current_gain > best_gain:
                best_gain = current_gain
                best_threshold = val

        return best_gain, best_threshold
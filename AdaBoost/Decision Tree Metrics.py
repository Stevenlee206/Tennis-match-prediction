import numpy as np

def entropy(labels : np.array)->float:
    num_samples=len(labels)
    _,counts=np.unique(labels,return_counts=True)
    prob_vector=counts/num_samples
    return -np.sum(prob_vector * np.log2(prob_vector + 1e-9))

def gain(labels : np.array,values : np.array,is_discrete : bool)->tuple:
    start_entropy=entropy(labels)
    n=len(labels)
    if is_discrete:
        unique_value=np.unique(values)
        weighted_entropy=0.0
        for value in unique_value:
            sub_labels=labels[values==value]
            weighted_entropy +=(len(sub_labels)/n)*entropy(sub_labels)
        gain=start_entropy-weighted_entropy
        return gain,None
    else:
        unique_value=np.unique(values)
        best_gain=0.0
        best_threshold=None
        for value in unique_value:
            left_branch=labels[values<value]
            right_branch=labels[values>=value]
            weighted_entropy=(len(left_branch)/n)*entropy(left_branch) + (len(right_branch)/n)*entropy(right_branch)
            gain=start_entropy-weighted_entropy
            if gain>best_gain:
                best_gain=gain
                best_threshold=value
        return best_gain,best_threshold


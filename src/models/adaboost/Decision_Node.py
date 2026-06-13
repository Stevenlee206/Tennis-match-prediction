class Decision_Node:
    def __init__(self,feature_idx=None,
                 threshold=None,is_discrete: bool = False,
                 child=None,label=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.is_discrete = is_discrete
        self.children = child if child is not None else {}
        self.label = label

    def is_leaf(self) -> bool:
        return self.label is not None
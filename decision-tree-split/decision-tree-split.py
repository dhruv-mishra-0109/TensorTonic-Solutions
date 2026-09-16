import numpy as np
def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    def gini_impurity(y):
        classes ,counts = np.unique(y,return_counts=True)
        class_probs = counts/len(y)
        gini = 1 - np.sum(class_probs**2)
        return gini
    best_feature_idx = -1
    best_threshold = 0
    best_information_gain = -np.inf
    parent_gini = gini_impurity(y)
    for feature_idx in range(X.shape[1]):
        X_fea = X[:,feature_idx]
        uniq_vals = np.unique(X_fea,sorted=True)
        for i in range(len(uniq_vals)-1):
            threshold = (uniq_vals[i]+uniq_vals[i+1])/2
            y_left = y[X_fea<=threshold]
            y_right = y[X_fea>threshold]
            info_gain = parent_gini-(len(y_left)*gini_impurity(y_left) + len(y_right)*gini_impurity(y_right))/len(y)
            if info_gain>best_information_gain:
                best_feature_idx = feature_idx
                best_threshold = threshold
                best_information_gain = info_gain
                
    return (best_feature_idx,best_threshold)
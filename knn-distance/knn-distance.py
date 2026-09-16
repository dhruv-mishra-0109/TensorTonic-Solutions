import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    # Write code here
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    if len(X_test)==0:
        return np.zeros((X_test.shape[0],int(k)))
    def euclidean_dist(x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    preds = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        dist = []
        for x2 in X_train:
            dist.append(euclidean_dist(x,x2))
        sorted_indices = list(np.argsort(np.array(dist))[:k])
        while len(sorted_indices)<k:
            sorted_indices.append(-1)
        preds.append(sorted_indices)
        
    return np.array(preds)
            
        
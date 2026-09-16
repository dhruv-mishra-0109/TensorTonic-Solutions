import math
import numpy as np

def gaussian_log_likelihood(x, mean=0, var=1):
    return -0.5 * np.log(2 * np.pi * var) - ((x - mean)**2) / (2 * var)

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes, counts = np.unique(y_train, return_counts=True)
    prior_probs = np.log(counts / len(y_train))
    means = {}
    vars_ = {}
    n_feature = X_train.shape[1]
    
    epsilon = 1e-9 * np.var(X_train, axis=0).max()
    if epsilon == 0:
        epsilon = 1e-2

    for clas in classes:
        X_clas = X_train[y_train == clas]
        means[clas] = []
        vars_[clas] = []
        
        for feature in range(n_feature):
            mean = np.mean(X_clas[:, feature])
            var = np.var(X_clas[:, feature]) + epsilon
            means[clas].append(mean)
            vars_[clas].append(var)

    predictions = []

    for x in X_test:
        class_pred = None
        prob_class = -np.inf
        for clas in classes:
            current_prob = prior_probs[np.argmax(classes == clas)]
            for feature_idx in range(X_test.shape[1]):
                mean = means[clas][feature_idx]
                var = vars_[clas][feature_idx]
                current_prob = current_prob + gaussian_log_likelihood(x[feature_idx], mean=mean, var=var)
            if current_prob > prob_class:
                class_pred = clas
                prob_class = current_prob
        predictions.append(class_pred)
    return predictions
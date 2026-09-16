import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    w = np.linalg.inv(X.T@X)@X.T@y
    return list(w)
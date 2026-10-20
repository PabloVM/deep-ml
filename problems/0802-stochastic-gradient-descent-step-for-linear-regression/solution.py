import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_samples = X.shape[0] 
    for i in range(n_iter):
            prediction = X[i % n_samples] @ weights
            error = prediction - y[i % n_samples]
            gradient = 2 * error * X[i % n_samples]
            weights -= gradient * learning_rate
    return weights

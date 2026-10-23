# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    lr = 0.01
    n_iters = 2000
    n = X.shape[0]
    
    for _ in range(n_iters):
        y_pred = X @ W + b
        residual = y_pred - y
        w_gradient = (2/n) * X.T @ residual
        b_gradient = (2/n) * np.sum(residual)
        
        W = W - w_gradient * lr
        b = b - b_gradient * lr
    
    return W,b
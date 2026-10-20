import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    batch_size = len(batch_indices)
    batched_X = np.array([X[i] for i in batch_indices])
    batched_X = np.hstack((batched_X,np.ones((batch_size,1))))
    batched_y = np.array([y[i] for i in batch_indices])
    weights = np.append(weights,bias)
    gradient = 2 / batch_size * batched_X.T @ (batched_X @ weights  - batched_y)
    
    weights -= gradient * lr

    
    return weights
import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        pred = X @ weights + bias
        error = pred - y
        w_grad = (1/n_samples) * X.T @ error + alpha1 * np.sign(weights) + 2 * alpha2 * weights
        b_grad = (1/n_samples) * np.sum(error)
        weights = weights - learning_rate * w_grad
        bias = bias - learning_rate * b_grad

        if np.linalg.norm(w_grad,1) < tol:
            break
    
    return weights.tolist(),float(bias)
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape
   
    
    match method:
        case "batch":
            for _ in range(n_epochs):
                gradient = 2 /  m  * X.T @ (X @ weights - y) 
                weights -= gradient * learning_rate
        case "stochastic":
            for _ in range(n_epochs):
                for i in range(m):
                    prediction = X[i] @ weights
                    error = prediction - y[i]
                    gradient = 2 * error * X[i]
                    weights -= gradient * learning_rate
        case "mini_batch":
            for _ in range(n_epochs):
                for i in range(0,m,batch_size):
                    gradient = 2 /  batch_size  * X[i : i + batch_size].T @ (X[i : i + batch_size] @ weights - y[i : i + batch_size]) 
                    weights -= gradient * learning_rate

    return weights

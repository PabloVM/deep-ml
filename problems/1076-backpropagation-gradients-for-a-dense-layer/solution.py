import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    z = W @ a_prev + b
    a = sigmoid(z)
    C = np.sum((a - y)**2)
    
    dz = 2 * (a - y) * a * (1-a)
    dW = dz[:,None] @ a_prev[None,:]
    db = dz
    da_prev = dz @ W

    return {
        "dW" : dW,
        "db" : db,
        "da_prev" : da_prev
    }
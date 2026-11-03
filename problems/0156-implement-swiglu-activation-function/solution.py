import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    d = x.shape[1] // 2
    x1 = x[:,:d]
    x2 = x[:,d:]

    swish_x2 = x2 / (1+np.exp(-x2))
    scores = x1 * swish_x2
    return scores
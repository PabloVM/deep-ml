import numpy as np

def max_backward(logits: np.ndarray, dlogit_maxes: np.ndarray) -> np.ndarray:
    """
    Backward pass for row-wise max.

    Args:
        logits: array of shape (N, C) -- input to the forward max
        dlogit_maxes: array of shape (N, 1) -- upstream gradient

    Returns:
        dlogits: array of shape (N, C) -- gradient w.r.t. logits
    """
    N,_ = logits.shape
    idx = np.argmax(logits,axis=1)
    dlogits = np.zeros_like(logits)
    dlogits[np.arange(N),idx] = dlogit_maxes[:,0]
    return dlogits

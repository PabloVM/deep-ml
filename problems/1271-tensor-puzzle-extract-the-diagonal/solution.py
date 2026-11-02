import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    n = A.shape[0]
    i = np.arange(n)

    return A[i,i]

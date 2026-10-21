import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    A = np.array(A, dtype=float)
    rows,columns = A.shape
    L = np.zeros_like(A, dtype=float)
    if not np.allclose(A,A.T):
        return -1
    if not np.all(np.linalg.eigvals(A) > 0):
        return -1
    
    for i in range(rows):
        for j in range(columns):
            if i== j:
                s = 0.0
                for k in range(i):
                    s += np.power(L[i][k], 2)
                L[i][i] = np.sqrt(A[i][i] - s)
            elif i > j:
                s=0.0
                for k in range(j):
                    s+=L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L.tolist()

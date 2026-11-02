import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    m,n = A.shape
    if m != n:
        return min(m,n)
    A = A.astype(float)
    for j in range(n):
        pivot =  j + np.argmax(np.abs(A[j:,j]), axis=0)
        if pivot != j:
            A[[j,pivot]] = A[[pivot,j]]
        for i in range(j+1,m):
            factor = A[i,j] / A[j,j]
            A[i,j:] -= factor * A[j,j:]
    
    rank = np.any(np.abs(A)>tol,axis=1).sum()

    return rank

import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    values = np.sort(np.real(np.linalg.eigvals(matrix)))
 
    return {
        "is_positive_definite" : bool(np.all(values > 1e-10)),
        "eigenvalues" : np.round(values,4).tolist()
    }
import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    values,_ = np.linalg.eigh(hessian)
    
    if np.any(values > tol) and np.any(values < -tol):
        return 0
    elif np.all(values > tol ):
        return -1
    elif np.all(values < -tol ):
        return +1
    else:
        return None

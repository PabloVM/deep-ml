import numpy as np


def find_leaky_features(X: np.ndarray, y: np.ndarray, threshold: float) -> list:
    """Column indices whose |correlation| with y is >= threshold.

    Args:
        X (np.ndarray): (n, p) feature matrix.
        y (np.ndarray): (n,) target.
        threshold (float): absolute-correlation cutoff.

    Returns:
        list[int]: ascending column indices.
    """
    X_centered = X - np.mean(X,axis=0)
    y_centered = y - np.mean(y)
    cov = np.mean((X_centered)*(y_centered[:,None]),axis=0)
    X_std = np.std(X,axis=0)
    X_std = np.where(X_std==0,1,X_std)
    y_std = np.std(y,axis=0)

    r = cov / (X_std * y_std)

    idx = np.where(np.abs(r)>threshold)[0]

    return idx.tolist()


def is_affine_function_of(col: np.ndarray, y: np.ndarray) -> bool:
    """True when col == a + b * y to within a residual of 1e-9."""
    # Your code here
    A = np.column_stack([np.ones(len(y)), y])
    a,b = np.linalg.lstsq(A,col,rcond=None)[0]

    residuals = col - (a + b * y)
    rss = np.sum(residuals ** 2)
    return bool(rss < 1e-9)


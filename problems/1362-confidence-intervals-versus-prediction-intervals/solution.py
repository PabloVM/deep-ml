import numpy as np


def interval_halfwidths(X: np.ndarray, y: np.ndarray, x0: np.ndarray, z: float) -> tuple:
    """Half-widths of the mean-response and new-observation intervals at x0.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.
        x0 (np.ndarray): (p,) point at which to form the intervals.
        z (float): multiplier, e.g. 1.96 for approximately 95%.

    Returns:
        tuple: (ci_half, pi_half) as floats.
    """
    n,p = X.shape
    D = np.column_stack([np.ones(len(X)),X])
    d = np.concatenate([[1.0], np.asarray(x0, dtype=float)])
    coef = np.linalg.lstsq(D,y,rcond=None)[0]
    rss = np.sum((y - D @ coef)**2)
    sigma2 = rss / (n - p -1)
    cov = sigma2 * d @ np.linalg.inv(D.T @ D) @ d
    ci_half = z * np.sqrt(cov)
    pi_half = z * np.sqrt( sigma2 + cov)

    return ci_half,pi_half


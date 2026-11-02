import numpy as np


def model_fit_stats(X: np.ndarray, y: np.ndarray) -> tuple:
    """Residual standard error, R-squared and the overall F-statistic.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (rse, r2, f_stat) as floats. RSS == 0 gives (0.0, 1.0, inf).
    """
    n,p = X.shape
    D = np.column_stack([np.ones(len(X)),X])
    coef = np.linalg.lstsq(D,y,rcond=None)[0]
    rss = np.sum((y - D @ coef)**2)
    tss = np.sum((y - np.mean(y))**2)
    if rss <= 1e-12 * tss:
        return (0.0,1.0,float("inf"))
    
    rse = np.sqrt(rss/(n-p-1))
    
    r_2 = 1 - (rss/tss)
    f = ((tss - rss) / p) / (rss / (n -p -1))

    return rse,r_2,f

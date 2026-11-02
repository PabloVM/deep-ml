import math

import numpy as np


def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares fit with an intercept, plus inference on each coefficient.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (coef, se, t, p), each a length-(p+1) array, intercept first.
    """
    n,p = X.shape
    D = np.column_stack([np.ones(len(X)),X])
    coef = np.linalg.lstsq(D,y,rcond=None)[0]

    resid = y - D  @ coef
    sigma2 = resid @ resid / (n - p -1)

    cov = sigma2 * np.linalg.inv(D.T @ D)
    se = np.sqrt(np.diag(cov))
    t = coef / se

    pval = np.array([2 * (1 - 0.5 * (1 + math.erf(abs(tj) / math.sqrt(2)))) for tj in t])

    return coef,se,t,pval
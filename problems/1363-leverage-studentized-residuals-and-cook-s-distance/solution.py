import numpy as np


def influence_measures(X: np.ndarray, y: np.ndarray) -> tuple:
    """Leverage, studentized residuals and Cook's distance for every observation.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (leverage, studentized, cooks), each a length-n array.
    """
    n,p = X.shape
    k= p + 1
    D = np.column_stack([np.ones(len(X)),X])
    coef = np.linalg.lstsq(D,y,rcond=None)[0]

    H = D @ np.linalg.inv(D.T @ D) @ D.T
    leverage = np.diag(H)

    resid = y - D  @ coef
    sigma2 = resid @ resid / (n - p -1)

    studentized = resid / (np.sqrt(sigma2) * np.sqrt(1- leverage))
    cooks = (studentized ** 2 / k) * (leverage / (1 - leverage))

    return leverage,studentized,cooks

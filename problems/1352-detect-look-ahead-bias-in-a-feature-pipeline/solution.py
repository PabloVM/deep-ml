import numpy as np


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score using only the trailing `window` values at each point.

    Args:
        x (np.ndarray): 1-D input series.
        window (int): lookback length, inclusive of the current point.

    Returns:
        np.ndarray: same length as x; the first window-1 entries are np.nan.
    """
    z_score = np.full(len(x),np.nan)

    for t in range(window - 1,len(x)):
        mean = np.mean(x[t-window+1 : t+1])
        std = np.std(x[t-window+1 : t+1])
        std = np.where(std==0,1,std)
        z_score[t] = (x[t] - mean)/std
    
    return z_score


def has_lookahead(feature_fn, x: np.ndarray) -> bool:
    """Return True if feature_fn uses information from the future."""
    base = feature_fn(x)
    x_copy = np.copy(x)
    x_copy[-1] += 1000.0
    new = feature_fn(x_copy)

    return np.all(base[:-1] != new[:-1])
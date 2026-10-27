# --- cell 1 ---
import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    nan_mean = np.nanmean(X_clean,axis=0)
    X_clean = np.where(np.isnan(X_clean),nan_mean,X_clean)
    
    return X_clean
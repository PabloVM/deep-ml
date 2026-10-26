import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    match strategy:
        case "mean":
            mean = np.nanmean(data,axis=0)
            result = np.where(np.isnan(data),mean,data)
        case "median":
            median = np.nanmedian(data,axis=0)
            result = np.where(np.isnan(data),median,data)
        case "mode":
            u, c = np.unique(data, return_counts=True,axis=0)
            mode = u[np.argmax(c, axis=0)]
            result = np.where(np.isnan(data),mode,data)
    
    return result
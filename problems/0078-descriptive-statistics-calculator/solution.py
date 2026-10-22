import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    data = np.array(data)
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]
    q1 = np.percentile(data,25)
    median = np.median(data)
    q3 = np.percentile(data,75)
    return{
        "mean" : np.mean(data),
        "median" : median,
        "mode" : mode,
        "variance" : np.var(data),
        "standard_deviation" : np.std(data),
        "25th_percentile": q1,
        "50th_percentile": median,
        "75th_percentile":q3,
        "interquartile_range":q3 - q1
    }
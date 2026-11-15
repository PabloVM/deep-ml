import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    if seed is not None:
        np.random.seed(seed)
    if mode == "fan_in":
        fan = n_in
    else:
        fan=n_out

    match distribution:
        case "normal":
            sigma = np.sqrt(2/fan)
            return np.random.randn(n_in,n_out) * sigma
        case "uniform":
            a = np.sqrt(6/fan)
            return np.random.uniform(-a,a,size=(n_in,n_out))
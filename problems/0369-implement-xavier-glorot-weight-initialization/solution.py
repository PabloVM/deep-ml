import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    if seed is not None:
        np.random.seed(seed)
    match mode:
        case "uniform":
            param = np.sqrt(6/(fan_in + fan_out))
            w = np.random.uniform(-param,param,(fan_in,fan_out))
        case "normal" : 
            param = np.sqrt(2/(fan_in + fan_out))
            w = np.random.randn(fan_in,fan_out) * param
        case _:
            raise ValueError("mode must be 'uniform' or 'normal'")
    return {
        "weights" : w,
        "shape" : [fan_in,fan_out],
        "param" : param
    }
import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    x_min = np.min(values)
    x_max = np.max(values)

    return c + ((d-c)/(x_max - x_min)) * (values - x_min)
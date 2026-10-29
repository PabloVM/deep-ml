import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    observations = np.array(observations)
    n = len(observations)
    k = np.sum(observations)

    #postirior

    a = alpha + k
    b = beta + n - k

    if a == 1 and b > 1:
        return 0
    elif a > 1 and b ==1:
        return 1

    return (a - 1) / (a + b -2)

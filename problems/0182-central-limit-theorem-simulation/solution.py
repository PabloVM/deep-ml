import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    match distribution:
        case "uniform":
            experiments = np.random.uniform(0, 1, size=(runs, n))
            mu = 0.5
            sigma = 1/12
        case "exponential":
            experiments = np.random.exponential(1.0, size=(runs, n))
            mu = 1
            sigma = 1
        case "bernoulli":
            experiments = (np.random.rand(runs, n) < 0.3).astype(float)
            mu = 0.3
            sigma = 0.21
        case _:
            raise ValueError()
    z = (np.mean(experiments,axis=1) - mu) / (np.sqrt(sigma) / np.sqrt(n))
    mean = np.mean(z)
    std = np.std(z)
    return {"mean":mean,"std":std}
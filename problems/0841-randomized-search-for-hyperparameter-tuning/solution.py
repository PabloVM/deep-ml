import numpy as np

def randomized_search(score_fn, param_grid, n_iter, seed):
    """
    Randomized hyperparameter search. Returns the best-scoring configuration as a dict
    with keys inserted in sorted order.
    """
    rng = np.random.default_rng(seed)
    asc_grid = {k: v for k, v in sorted(param_grid.items(), key=lambda item: item[0])}
    best_score = float("-inf")
    best_params = None
    for _ in range(n_iter):
        params = {}
        for k in asc_grid.keys():
            idx = rng.integers(0,len(asc_grid[k]))
            params[k] = asc_grid[k][idx]
        score = score_fn(**params)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params

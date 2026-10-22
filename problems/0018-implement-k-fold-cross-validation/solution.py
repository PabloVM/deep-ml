import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    idx = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idx)

    fold_size = n_samples // k

    splits = []

    start = 0

    for fold in range(k):
        size = n_samples // k + (1 if fold < n_samples % k else 0)
        end = start + size
        test = idx[start:end]
        train = np.concatenate([idx[:start],idx[end:]])

        splits.append((train,test))
        start = end

    return splits

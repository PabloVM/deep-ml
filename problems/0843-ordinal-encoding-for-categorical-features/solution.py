import numpy as np

def ordinal_encode(X, categories):
    """
    Encode categorical features as integer codes using a provided ordering.

    Args:
        X: 2D list/array of shape (n_samples, n_features) with string values.
        categories: list of length n_features; categories[j] is an ordered
                    list of valid categories for column j.

    Returns:
        np.ndarray of shape (n_samples, n_features), dtype int.
        Unknown categories are encoded as -1.
    """
    encode_dicts = [{cat:code for code,cat in enumerate(cats)} for cats in categories]
    encoded_X = [[encode_dicts[i].get(cat,-1) for i,cat in enumerate(row)] for row in X]
    return np.array(encoded_X)




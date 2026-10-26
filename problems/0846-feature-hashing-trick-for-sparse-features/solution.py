import hashlib

def feature_hash(features, n_features):
    """
    Apply the (signed) feature hashing trick.

    Args:
        features: dict mapping feature name (str) -> value (float).
        n_features: int, size of the output vector.

    Returns:
        list of length n_features.
    """
    result = [0 for _ in range(n_features)]
    for name,value in features.items():
        md5 = hashlib.md5(name.encode("utf-8"))
        h = md5.hexdigest()
        idx = int(h[0:8],16) % n_features
        sign = 1 if int(h[8:16],16) % 2 == 0 else -1
        result[idx] = sign * value
    
    return result
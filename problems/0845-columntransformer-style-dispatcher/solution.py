import numpy as np

def column_transformer(X, transformers):
    """
    Apply per-column-group transformations and horizontally concatenate the results.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        transformers: list of (name, kind, columns) tuples

    Returns:
        2D list of floats with transformed and concatenated columns.
    """
    X = np.array(X,dtype=float)
    output = []

    for step in transformers:
        trans = step[1]
        columns = step[2]

        match trans:
            case "standard":
                mean = np.mean(X[:,columns],axis=0)
                std = np.std(X[:,columns],axis=0)
                std = np.where(std==0,1,std)
                output.append((X[:,columns]- mean) / std)
            case "minmax":
                min_v = np.min(X[:,columns],axis=0)
                max_v = np.max(X[:,columns],axis=0)
                max_min = max_v - min_v
                max_min = np.where(max_min==0,1,max_min)
                output.append((X[:,columns] - min_v) / max_min)
            case "passthrough":
                output.append(X[:,columns])
            case "drop":
                pass
    if len(output)>0:
        return np.hstack(output).tolist() 
    else:
        return []

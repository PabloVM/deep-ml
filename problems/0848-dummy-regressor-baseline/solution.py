import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    y_train = np.array(y_train)


    match strategy:
        case "mean":
            result = np.mean(y_train)
        case "median":
            result = np.median(y_train)
        case "quantile":
            if quantile is None or quantile < 0 or quantile > 1:
                raise ValueError("quantile must be in range [0,1]")
            result = np.quantile(y_train,quantile)
        case "constant":
            if constant is None:
                raise ValueError("constant can't  be None")
            result = constant
    return [result for _ in range(n_test)]



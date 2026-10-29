import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.array(predictions)
    y_true = np.array(y_true)
    m_pred = np.mean(predictions,axis=0)
    s_bias = np.mean((m_pred - y_true)**2)
    v = np.mean(np.mean((predictions - m_pred)**2,axis=0))
    mse = np.mean((predictions- y_true)**2)
    return{
        "bias_squared" : float(s_bias),
        "variance" : float(v),
        "mse" : float(mse)
    }

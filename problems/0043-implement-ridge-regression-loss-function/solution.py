import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	return (1/X.shape[0]) * np.sum((y_true - X @ w)**2) + alpha * np.sum(w**2)

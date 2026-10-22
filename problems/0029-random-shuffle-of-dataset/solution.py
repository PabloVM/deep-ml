import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed is not None:
		np.random.seed(seed)
	shuffled_idx = np.random.permutation(np.arange(X.shape[0]))

	return (X[shuffled_idx],y[shuffled_idx])

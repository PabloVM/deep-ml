import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	if y is None:
		return [X[i:i+batch_size] for i in range(0,X.shape[0],batch_size)]
	
	return [[X[i:i+batch_size],y[i:i+batch_size]] for i in range(0,X.shape[0],batch_size)]
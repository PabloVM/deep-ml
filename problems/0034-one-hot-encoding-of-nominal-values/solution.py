import numpy as np

def to_categorical(x, n_col=None):
	x = np.array(x)
	unique_classes = max(len(np.unique(x)),n_col or 0)
	one_hot = np.zeros((x.shape[0],unique_classes))
	for i,element in enumerate(x):
		one_hot[i,element] = 1
	
	return one_hot.tolist()
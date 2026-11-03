import numpy as np

def square_relu(x: np.ndarray) -> dict:
	"""
	Apply the Square ReLU activation function and compute its derivative.
	
	Args:
		x: Input numpy array of any shape
	
	Returns:
		Dictionary with 'output' and 'derivative' as numpy arrays
	"""
	output = np.maximum(0,x)**2
	derivative = np.maximum(0,x) * 2
	return{
		"output":np.round(output,2),
		"derivative":np.round(derivative,2)
	}
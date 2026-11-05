import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	f_x = f(x)
	f_x_h = [np.array(f([x[n] + h if n==m else x[n] for n in range(len(x))])).T for m in range(len(x))]
	f_x = np.array(f_x)
	f_x_h = np.array(f_x_h)
	j =( (f_x_h - f_x) / h).T
	return j.tolist()
			

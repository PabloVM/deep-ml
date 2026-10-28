from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	n=len(point)
	hessian = [[0.0 for _ in range(n)]for _ in range(n)]

	def move_point(point, i=None, di = 0.0, j=None, dj=0.0):
		new_point = list(point)
		if i is not None:
			new_point[i] += di
		if j is not None:
			new_point[j] += dj
		return new_point

	f_x = f(point)

	for i in range(n):
		for j in range(i + 1):
			if i == j:
				hessian[i][j] = (f(move_point(point,i=i,di=h)) - 2 * f_x +f(move_point(point,i=i,di=-h))) / h**2
			else:
				value = (f(move_point(point,i,h,j,h)) - f(move_point(point,i,h,j,-h)) - f(move_point(point,i,-h,j,h)) + f(move_point(point,i,-h,j,-h))) / (4 * h ** 2)
				hessian[i][j] = value
				hessian[j][i] = value

	
	return hessian
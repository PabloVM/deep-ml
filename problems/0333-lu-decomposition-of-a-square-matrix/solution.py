import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	n = len(A)
	L = np.eye(n)
	U = np.zeros_like(L)
	for i in range(n):
		for j in range(i,n):
			s = 0.0
			for k in range(n):
				s+= L[i][k] * U[k][j]
			U[i][j] = A[i][j] -s
		for j in range(i+1,n):
			s = 0.0
			for k in range(i):
				s+= L[j][k] * U[k][i]
			L[j][i] = (A[j][i] - s) / U[i][i]

	return(L,U)
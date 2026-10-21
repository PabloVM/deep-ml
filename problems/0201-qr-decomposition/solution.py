import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	q = []
	A = np.array(A)

	r,c = A.shape

	q.append(A[:,0]/ np.linalg.norm(A[:,0]))
	for i in range(1,c):
		s = 0
		for j in range(i):
			s += q[j] *( A[:,i] @ q[j]) / np.linalg.norm(q[j])
		v = A[:,i] - s
		q.append(v / np.linalg.norm(v))
	
	
	Q = np.column_stack(q)
	R = Q.T @ A

	return(Q,R)
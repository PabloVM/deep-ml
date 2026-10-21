import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	x = x0 if x0 != None else np.zeros_like(b)
	r = b - A @ x
	p = r

	it = 0

	while it < n:
		alpha = (r.T @ r) /( p.T @ A @ p)
		x = x + alpha * p
		rn = r - alpha * A @ p

		if np.abs(np.linalg.norm(r)) < tol:
			break
		
		beta = (rn.T @ rn) / (r.T @ r)

		p = rn + beta * p
		r = rn
		it +=1

	return x

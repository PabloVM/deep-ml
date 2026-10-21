import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	Ab = np.column_stack((A,b))
	rows,columns = Ab.shape
	solution = np.zeros(rows)
	for k in range(rows):
		max_element = np.abs(Ab[k,k])
		best_row = k
		for i in range(k+1,rows):
			element = Ab[i,k]
			if np.abs(element) > max_element:
				max_element = element
				best_row = i
		Ab[[k,best_row]] = Ab[[best_row,k]]
		for i in range(k+1,rows):
			factor = Ab[i,k]/Ab[k,k]
			Ab[i,k:] -= factor * Ab[k,k:]
	
	
	for k in reversed(range(rows)):
		x = Ab[k,-1]
		for i in range(k+1,rows):
			x-= Ab[k,i] * solution[i]
		solution[k] = x / Ab[k,k]
	return solution

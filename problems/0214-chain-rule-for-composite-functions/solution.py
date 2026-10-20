import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	functions_dict = {
		"square" : lambda x: np.power(x, 2),
		"sin" : lambda x: np.sin(x),
		"exp" : lambda x: np.exp(x),
		"log" : lambda x:  np.log(x)
	}

	derivated_functions_dict = {
		"square" : lambda x: 2 * x,
		"sin" : lambda x: np.cos(x),
		"exp" : lambda x: np.exp(x),
		"log" : lambda x:  1 / x
	}
	
	result = 1.0
	for f in reversed(functions):
		result *= derivated_functions_dict[f](x)
		x = functions_dict[f](x)
	return result


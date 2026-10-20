import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	gradient = np.array(gradient)
	magnitude = np.sqrt(np.sum(np.power(gradient, 2)))
	direction = gradient / magnitude if magnitude != 0 else np.zeros_like(gradient)
	return {
		"magnitude" : magnitude ,
		"direction" : direction,
		"descent_direction" : -  direction
	}
def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	updated = []
	for params,grads,apply in zip(parameters,gradients,apply_to_all):
		wd = weight_decay if apply else 0
		new_group = [(1-lr * wd) * w - lr * g for w,g in zip(params,grads)]
		updated.append(new_group)
	return updated

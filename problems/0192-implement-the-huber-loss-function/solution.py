def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	if isinstance(y_true, (int, float)):
    	y_true, y_pred = [y_true], [y_pred]
		
	errors = [abs(true - pred) for true,pred in zip(y_true,y_pred)]
	huber_losses = []
	for i,error in enumerate(errors):
		if error <= delta:
			huber_losses.append(0.5*(y_true[i] - y_pred[i])**2)
		else:
			huber_losses.append(delta * (abs(y_true[i] - y_pred[i]) - 0.5 * delta))
	
	return sum(huber_losses)/len(huber_losses)
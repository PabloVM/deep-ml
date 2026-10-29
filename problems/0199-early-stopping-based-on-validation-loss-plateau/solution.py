def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
	"""
	Determine at each epoch whether training should stop based on validation loss.
	
	Args:
		val_losses: List of validation losses at each epoch
		patience: Number of epochs to wait for improvement before stopping
		min_delta: Minimum change in validation loss to qualify as improvement
	
	Returns:
		List of booleans indicating whether to stop at each epoch
	"""

	best_loss = float('inf')
	epochs_without_improvement = 0

	epoch_stop = [False for _ in range(len(val_losses))]

	for i in range(len(val_losses)):
		if val_losses[i] < best_loss - min_delta:
			epochs_without_improvement = 0
			best_loss = val_losses[i]
		else:
			epochs_without_improvement +=1

		if epochs_without_improvement >= patience:
			epoch_stop[i] = True
			

	return epoch_stop

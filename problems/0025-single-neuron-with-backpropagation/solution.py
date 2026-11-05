import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	W = initial_weights
	b = initial_bias
	n = len(features)
	mse_values = []
	for epoch in range(epochs):
		z = features @ W + b
		s = sigmoid(z)
		mse_values.append(np.mean((s-labels)**2))

		w_grad = (2/n) * (((s - labels) * (s * (1-s))) @ features)
		b_grad = (2/n) * np.sum((s - labels) * (s * (1-s)))

		W -= learning_rate * w_grad
		b -= learning_rate * b_grad

	updated_weights = W 
	updated_bias = b

	return updated_weights, updated_bias, mse_values
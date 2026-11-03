import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	z = [sum([features[i][j] * weights[j] for j in range(len(weights))]) + bias for i in range(len(features))]
	probabilities = [ 1 / (1 + math.exp(-z[i])) for i in range(len(z))]
	mse = sum([(probabilities[i] - labels[i])**2 for i in range(len(probabilities))]) / len(probabilities)
	return probabilities, mse
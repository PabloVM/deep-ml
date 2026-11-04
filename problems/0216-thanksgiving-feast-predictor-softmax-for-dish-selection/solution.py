import math
def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
	"""
	Predict the probability of choosing each Thanksgiving dish using softmax.
	
	Args:
		preference_scores: List of preference scores for each dish
		(e.g., [turkey_score, stuffing_score, cranberry_score, pie_score])
		
	Returns:
		List of probabilities for each dish
	"""
	max_score = max(preference_scores)
	exponentials = [math.exp(score - max_score) for score in preference_scores]
	exponentials_sums = sum(exponentials)
	return [exp/exponentials_sums for exp in exponentials]
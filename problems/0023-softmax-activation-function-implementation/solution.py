import math

def softmax(scores: list[float]) -> list[float]:
    z_max = max(scores)
    z_sum = sum([math.exp(score - z_max) for score in scores])
    return [math.exp(score - z_max) / z_sum for score in scores]
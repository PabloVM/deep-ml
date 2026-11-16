import numpy as np


def activation_health(acts, threshold=0.97):
    """One dict per layer: mean, std, saturated (% of values with |t| > threshold) and dead (neuron indices)."""
    layers = []
    for act in acts:
        mean = round(float(np.mean(act)),4)
        std = round(float(np.std(act)),4)
        saturated = round(float(np.sum(np.where(np.abs(act) > threshold,1,0)) / act.size  * 100),2)
        dead = np.nonzero(np.all(np.abs(act) > threshold,axis=0))
        layers.append(
            {
                "mean":mean,
                "std":std,
                "saturated":saturated,
                "dead":dead[0].tolist()
            }
        )
    return layers

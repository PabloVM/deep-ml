import numpy as np

def he_initialize(layer_dims: list, method: str = 'normal', seed: int = 42) -> list:
    """
    Initialize weight matrices for a neural network using He initialization.
    
    Args:
        layer_dims: List of integers representing neurons per layer
        method: 'normal' or 'uniform' sampling distribution
        seed: Random seed for reproducibility
    
    Returns:
        List of numpy arrays, one weight matrix per adjacent layer pair
    """
    np.random.seed(seed)
    w = []
    match method:
        case "normal":
            for layer in range(len(layer_dims) - 1):
                param = np.sqrt(2/layer_dims[layer])
                w.append(np.random.randn(layer_dims[layer],layer_dims[layer + 1]) * param)
        case "uniform":
            for layer in range(len(layer_dims) -1):
                param = np.sqrt(6/layer_dims[layer])
                w.append(np.random.uniform(-param,param,size=(layer_dims[layer],layer_dims[layer + 1])))
    return w


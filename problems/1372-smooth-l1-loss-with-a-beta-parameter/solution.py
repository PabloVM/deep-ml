import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):
    """
    Elementwise Smooth L1 loss.

    Args:
        pred, target: arrays of the same shape
        beta: transition point between the quadratic and linear branches (> 0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        float for 'mean'/'sum', array for 'none'
    """
    reductions = {
        "mean" : lambda x: float(np.mean(x)),
        "sum"  : lambda x: float(np.sum(x)),
        "none" : lambda x: x.tolist()
    }
    if reduction not in reductions.keys() or beta <= 0:
        raise ValueError("Wrong input parameters")

    pred = np.array(pred)
    target = np.array(target)
    error =  pred - target

    loss = np.where(np.abs(error) < beta , (0.5 * error ** 2) / beta , np.abs(error) - 0.5 * beta)

    
    
    return reductions[reduction](loss)

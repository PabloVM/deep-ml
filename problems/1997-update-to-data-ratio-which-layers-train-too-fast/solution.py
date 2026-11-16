import numpy as np


def update_ratios(params, grads, lr, low=-4.0, high=-2.0):
    """For every 2-D parameter: {"index", "ratio" = log10(std(lr * grad) / std(param)), "status"}."""
    ratios = []
    for i,(param,grad) in enumerate(zip(params,grads)):
        if param.ndim != 2:
            continue
        param_std = np.std(param)
        if param_std == 0:
            ratio = None
            status ="undefined"
        else:
            ratio =  round(float(np.log10(np.std(lr * grad) / param_std)),3)
            if ratio < low:
                status = "slow"
            elif ratio > high:
                status = "fast"
            else:
                status = "ok"
        ratios.append({
            "index":i,
            "ratio":ratio,
            "status":status
        })
    return ratios

def lr_multiplier(ratios, target=-3.0):
    """The factor to multiply the learning rate by so the median ratio lands on target."""
    ratios = [ratio for ratio in ratios if ratio is not None]
    r_mean = np.median(ratios)
    return 10 ** (target-r_mean)

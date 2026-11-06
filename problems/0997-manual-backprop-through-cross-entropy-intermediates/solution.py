import numpy as np

def manual_cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.

    Args:
        logits: numpy array of shape (n, C)
        y:      numpy array of integer class indices, shape (n,)

    Returns:
        Dictionary with keys 'dlogprobs', 'dprobs', 'dcounts_sum_inv',
        'dcounts_sum', 'dcounts', 'dnorm_logits', 'dlogit_maxes', 'dlogits'.
    """
    n,C = logits.shape
    logits_maxes = logits.max(axis=1,keepdims=True)
    norm_logits = logits - logits_maxes
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(axis=1,keepdims=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    log_probs = np.log(probs)
    loss = -np.mean(log_probs[np.arange(n),y])

    dlog_probs = np.zeros((n,C))
    dlog_probs[np.arange(n),y] = -1/n
    dprobs = dlog_probs * (1/probs)
    dcounts_sum_inv = np.sum(dprobs * counts,axis=1,keepdims=True)
    dcounts_sum = -counts_sum ** -2 * dcounts_sum_inv
    dcounts = dprobs * counts_sum_inv + dcounts_sum
    dnorm_logits = dcounts * counts
    dlogit_maxes = - np.sum(dnorm_logits,axis=1,keepdims=True)
    temp_zeros = np.zeros((n,C))
    temp_zeros[np.arange(n),logits.argmax(axis=1)] = 1
    dlogits = dnorm_logits + temp_zeros * dlogit_maxes

    return{
        'dlogprobs':dlog_probs,
        'dprobs':dprobs,
        'dcounts_sum_inv':dcounts_sum_inv,
        'dcounts_sum':dcounts_sum,
        'dcounts':dcounts,
        'dnorm_logits':dnorm_logits,
        'dlogit_maxes':dlogit_maxes,
        'dlogits':dlogits
    }

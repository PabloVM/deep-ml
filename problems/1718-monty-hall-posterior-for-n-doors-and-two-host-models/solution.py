import numpy as np
import math
def monty_posterior(n_doors, chosen, opened, host):
    # host in {"knows", "random"}; doors numbered from 0
    # return list of n_doors posterior probabilities
    def comb(n,r):
        return math.factorial(n) / (math.factorial(r) * math.factorial(n-r))
    p_h_i = 1 / n_doors
    k = len(opened)
    available = n_doors - 2
    match host:
        case "knows":
            likelihood = [0.0 if i in opened else  1 / comb(available,k) if i!= chosen else  1/ comb(available + 1,k) for i in range(n_doors)]
        case "random":
            likelihood = [0.0 if i in opened else 1/(n_doors - k) for i in range(n_doors)]
    
    posterior = np.array(likelihood) * p_h_i
    posterior_sum = np.sum(posterior)

    return (posterior/posterior_sum).tolist()
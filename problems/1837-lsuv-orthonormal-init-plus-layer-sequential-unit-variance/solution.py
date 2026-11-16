import numpy as np

def lsuv_init(layer_dims, X_batch, act, seed, tol=0.1, max_iter=10):
    # return (Ws, variances, n_rescales)
    rng = np.random.default_rng(seed)
    w = []

    match act:
        case "relu":
            f = lambda x : np.maximum(0,x)
        case "tanh":
            f = lambda x : np.tanh(x)
        case "linear":
            f = lambda x : x 

    for layer in range(len(layer_dims) - 1):
        d_l = layer_dims[layer+1]
        d_lprev = layer_dims[layer]
        G = rng.standard_normal((d_l,d_lprev))
        if d_l >= d_lprev:
            Q,R = np.linalg.qr(G)
            w.append(Q @ np.diag(np.sign(np.diag(R))))
        else:
            Q,R = np.linalg.qr(G.T)
            w.append((Q @ np.diag(np.sign(np.diag(R)))).T)
        
    variances = []
    n_rescales = []
    h_in = X_batch
    for layer in range(len(layer_dims)-1):
        i = 0
        z = h_in @ w[layer].T
        h_out = f(z)
        v = np.var(z)
        while np.abs(v-1) >= tol and i < max_iter:
            w[layer] = w[layer] / np.sqrt(v)
            z = h_in @ w[layer].T
            h_out = f(z)
            v = np.var(z)
            i+=1
        h_in = h_out
        variances.append(v)
        n_rescales.append(i)
    
    return w,variances,n_rescales






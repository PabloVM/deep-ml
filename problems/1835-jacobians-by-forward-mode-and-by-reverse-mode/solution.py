import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def activation(act,derivative,x):
        match act:
            case "relu":
                if derivative:
                    return np.where(x>0.0,1.0,0.0)
                else:
                    return np.where(x>0.0,x,0.0)
            case "tanh":
                if derivative:
                    return 1 - np.tanh(x) ** 2
                else:
                    return np.tanh(x)
            case "sigmoid":
                if derivative:
                    return sigmoid(x) * (1-sigmoid(x))
                else:
                    return sigmoid(x)
            case "linear":
                if derivative:
                    return np.ones_like(x)
                else:
                    return x


def jacobian_two_ways(layers, x):
    # layers: list of (W, b, act); return dict(J_fwd, J_rev, n_jvp, n_vjp, macs_fwd, macs_rev)
    derivatives = []
    output = x

    for W,b,act in layers:
        output = W @ output + b
        derivatives.append(activation(act,True,output))
        output = activation(act,False,output)
    
    #Forward Mode
    n_jvp = len(x)
    j_fwd = np.zeros((len(output),n_jvp))
    I = np.eye(n_jvp)
    
    for i in range(n_jvp):
        e = I[:,i]
        for layer,act_der in zip(layers,derivatives):
            w = layer[0]
            e = (w @ e) * act_der
        j_fwd[:,i] = e
   
    #Reverse Mode
    n_vjp = len(output)
    j_rev = np.zeros((n_vjp,len(x)))
    I = np.eye(n_vjp)

    for i in range(n_vjp):
        u = I[i,:]
        r_layers = reversed(layers)
        r_derivatives = reversed(derivatives)
        for layer,act_der in zip(r_layers,r_derivatives):
            w = layer[0]
            u = w.T @ ( act_der * u)
        j_rev[i,:] = u

    w_cost = sum([w.shape[0] * w.shape[1] for w,_,_ in layers])
    macs_fwd = w_cost * n_jvp
    macs_rev =  w_cost * n_vjp

    return {
        "J_fwd" : j_fwd,
        "J_rev" : j_rev,
        "n_jvp" : n_jvp,
        "n_vjp" : n_vjp,
        "macs_fwd" : macs_fwd,
        "macs_rev" : macs_rev
    }



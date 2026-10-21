import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    mse = []
    n,d = X.shape
    weights = w0
    for k in range(n_steps):
        if n_steps == 1:
            lr = 10 ** a
        else:
            lr = 10 ** (a +(b-a) * k /( n_steps -1))
        
        
        gradient = 2 / n * X.T @(X @ weights - y)
        weights -= lr * gradient

        loss =np.mean(( X @ weights - y)**2)
        
        mse.append(float(loss))
    
    return mse

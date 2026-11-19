import numpy as np

def newton_step(g, H):
    return np.linalg.solve(H,-g)

def damped_step(g, H, lam):
    D = H.shape[0]
    return -np.linalg.inv(H + lam * np.eye(D)) @ g

def trust_region_minimize(f, grad, hess, x0, lam0=1.0, n_iter=100, tol=1e-8):
    
    lam = lam0
    x=x0
    for i in range(n_iter):
        g = grad(x)
        if np.linalg.norm(g,ord=2) < tol:
            return x,i
        H = hess(x)
        D = H.shape[0] 
        while np.min(np.linalg.eig(H + lam * np.eye(D))[0]) <= 10e-12:
            lam = lam * 10
        step = damped_step(g,H,lam)
        pred = -(g.T @ step + 0.5 * step.T @ H @ step)
        act = f(x) - f(x+step)
        ratio = act/pred
        if ratio > 0.75:
            lam = np.maximum(lam/3,10e-12)
        elif ratio < 0.25:
            lam = 4 * lam
        if ratio > 0:
            x = x + step
    return x,n_iter


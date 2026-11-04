import numpy as np
def sigmoid(x):
     return 1 / (1 + np.exp(-x)) 

def rbf_design(X, centers, sigma):
    d = (X[:,None,:] - centers[None,:,:])
    squared_distance = np.sum(d**2,axis=-1)
    activations = np.exp(-1/(2*sigma**2) * squared_distance)
    design_matrix = np.hstack([np.ones((activations.shape[0],1)),activations])
    return design_matrix

def rbf_fit(X, y, centers, sigma, lam, task, n_newton=20):



    k = centers.shape[0]
    d = centers.shape[1]
    phi = rbf_design(X,centers,sigma)
    D = np.eye(k+1)
    D[0,0] = 0.0
    match task:
        case "reg":
            w = np.linalg.solve(phi.T @ phi + lam * D,phi.T @ y)
            return w
        case "clf":
            w = np.zeros(k+1)
            for _ in range(n_newton):
                p = sigmoid(phi @ w)
                g = phi.T @ (p - y) + lam * D @ w
                H = phi.T @ np.diag(p*(1-p)) @ phi + lam * D
                H_1 = np.linalg.inv(H)
                w -= H_1 @ g
            return w

def rbf_predict(X, centers, sigma, w, task):
    phi = rbf_design(X,centers,sigma)
    match task:
        case "reg":
            return phi @ w
        case "clf":
            return sigmoid(phi @ w)


import numpy as np

def exact_step(A, b, theta, d):
    g = A @ theta + b
    return -(d.T @ g) / (d.T @ A @ d)
def armijo_backtrack(f, grad, theta, d, eta0=1.0, c=1e-4, rho=0.5, max_halvings=60):
    eta = eta0
    for i in range(max_halvings):
        l_armijo = f(theta + eta * d)
        r_armijo = f(theta) + c * eta * d.T @ grad(theta)
        if l_armijo <= r_armijo:
            return eta
        if i == max_halvings:
            return eta
        
        eta *= rho

def steepest_descent(f, grad, theta0, n_iter, rule="armijo", eta=0.1, A=None, b=None):
    # return iterates, shape (n_iter + 1, D)
    thetas=[theta0]
    theta = theta0.copy()
    for i in range(n_iter):
        d = -grad(theta)
        match rule:
            case "fixed":
                pass
            case "exact":
                eta = exact_step(A,b,theta,d)
            case "armijo":
                eta = armijo_backtrack(f,grad,theta,d)
        theta = theta + eta * d
        thetas.append(theta)
    return np.array(thetas)


import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    num_g = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        g_i = (f(x_plus) - f(x_minus)) / (2*epsilon)
        num_g[i] = g_i
    r_e = np.linalg.norm(num_g - analytical_grad) / (np.linalg.norm(num_g) + np.linalg.norm(analytical_grad))
    return (num_g,r_e)
import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix
        c: 2-element vector (linear coefficients)
        a: 2-element vector (constraint coefficients)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    kkt_left = np.block([
        [Q,-a[:,None]],
        [a,0]
        ])
    kkt_right = np.append(-c,b)

    x1,x2,l = np.linalg.solve(kkt_left,kkt_right)
    x = np.array([x1,x2])
    obj = 0.5 * x.T @ Q @ x + c @ x
    
    return{
        "x" : [round(float(x1),4),round(float(x2),4)],
        "lambda": round(float(l),4),
        "objective" : round(float(obj),4)
    }
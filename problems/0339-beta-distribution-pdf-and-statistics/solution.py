import math

def beta_distribution_stats(x: float, alpha: float, beta_param: float) -> dict:
    """
    Compute Beta distribution statistics.
    
    Args:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (alpha > 0)
        beta_param: Second shape parameter (beta > 0)
    
    Returns:
        Dictionary with 'pdf', 'mean', and 'variance'
    """
    if x<0 or x>1:
        pdf_x = 0.0
    else:
        beta_f = (math.gamma(alpha) * math.gamma(beta_param)) / math.gamma(alpha + beta_param)
        pdf_x = (math.pow(x,alpha-1) * math.pow(1-x,beta_param-1)) / beta_f
    mean = alpha / (alpha + beta_param)
    variance = (alpha * beta_param) / ((alpha + beta_param) **2 * (alpha+beta_param+1))

    return{
        "pdf": pdf_x,
        "mean":mean,
        "variance":variance
    }
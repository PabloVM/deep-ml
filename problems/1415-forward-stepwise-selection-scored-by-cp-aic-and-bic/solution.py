import numpy as np


def rss(X, y, cols):
    # TODO: OLS with intercept on X[:, cols]; return the residual sum of squares
    A = np.column_stack([np.ones(X.shape[0]), X[:, cols]])
    w = np.linalg.lstsq(A,y)[0]
    y_pred = A @ w
    residuals = y_pred - y
    return np.sum(residuals**2)


def forward_stepwise(X, y):
    # TODO: greedy path; models[k] = indices chosen after k steps, in order of addition
    models = [[]]
    p = X.shape[1]

    for _ in range(p):
        actual = models[-1]
        columns = np.setdiff1d(np.arange(p),actual)
        min_rss = float("inf")
        best_column = None
        for column in columns:
            r = rss(X,y,actual + [column])
            if   r < min_rss:
                min_rss = r
                best_column = column
        models.append(actual + [int(best_column)])
    
    return models



def cp_aic_bic(rss_value, n, d, sigma2):
    # TODO: (Cp, AIC, BIC) as defined in the statement, rounded to 4 decimals
    Cp = np.round((rss_value + 2 * d * sigma2) / n,4)
    AIC = np.round((rss_value + 2 * d * sigma2) / (n * sigma2),4)
    BIC = np.round((rss_value + np.log(n)*d *sigma2) / n,4)

    return float(Cp),float(AIC),float(BIC)


def select_by_bic(X, y):
    # TODO: sigma2 from the full model, BIC along the path, return (sorted best cols, bics)
    n,p = X.shape
    models = forward_stepwise(X,y)
    sigma2 = rss(X,y,models[-1]) / (n - p -1)

    best_model = None
    lowest_bic = float("inf")
    bics = []
    for model in models:
        r = rss(X,y,model)
        _,_,bic = cp_aic_bic(r,n,len(model),sigma2)
        bics.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_model = model
    
    return sorted(best_model),bics


import numpy as np


def ridge_fit(X, y, lam):
    # TODO: center X and y, solve the penalized normal equations, recover the intercept; return (w, b)
    X = np.array(X)
    y = np.array(y)

    x_mean = np.mean(X,axis=0)
    y_mean = np.mean(y,axis=0)
    X = X - x_mean
    y = y - y_mean
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]),X.T @ y)
    b = y_mean - x_mean @ w
    return w,b


def kfold_mse(X, y, lam, k, seed=0):
    # TODO: seeded permutation, array_split into k folds, per-fold MSE; return (mean, se)
    n,m = X.shape
    idx = np.arange(n)
    idx = np.random.default_rng(seed).permutation(idx)
    splits = np.array_split(idx,k)

    mse = []

    for split in splits:
        train_idx = np.setdiff1d(idx,split)
        X_train =  X[train_idx]
        y_train = y[train_idx]
        X_val = X[split]
        y_val = y[split]

        w,b = ridge_fit(X_train,y_train,lam)
        y_pred = X_val @ w + b
        residual = y_pred - y_val

        mse.append(np.mean(residual**2))
    
    return float(np.mean(mse)) , float(np.std(mse,ddof=1) / np.sqrt(k))


def select_lambda(X, y, lambdas, k, seed=0):
    # TODO: curve over lambdas; lam_min; lam_1se = largest lambda with mean <= mean_min + se_min
    means,ses = [],[]

    for lam in lambdas:
        mean,se = kfold_mse(X,y,lam,k,seed)
        means.append(mean)
        ses.append(se)

    min_idx = np.argmin(means)
    lam_min = lambdas[min_idx]
    mean_min = means[min_idx]
    se_min = ses[min_idx]

    threshold = mean_min + se_min

    lam_1se = float("-inf")
    for lam,mean in zip(lambdas,means):
        if mean <= threshold and lam > lam_1se:
            lam_1se = lam
    
    return lam_min,lam_1se,[round(float(m),4) for m in means]








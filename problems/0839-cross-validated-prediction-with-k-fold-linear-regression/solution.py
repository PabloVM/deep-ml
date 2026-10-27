import numpy as np

def cross_val_predict(X, y, k):
    """
    Return out-of-fold predictions for every sample using k-fold
    cross-validation with a linear regression model (with intercept).
    """
    X = np.array(X)
    y = np.array(y)
    rows,columns = X.shape
    idx = np.arange(rows)

    splits = []

    start = 0

    for fold in range(k):
        size = rows // k + (1 if fold < rows % k else 0)
        end = start + size
        val = idx[start:end]
        train = np.concatenate([idx[:start],idx[end:]])
        splits.append((train,val))
        start = end

    predictions = np.empty(rows,dtype=float)

    for train_idx,val_idx in splits:
        X_train = X[train_idx]
        m,n = X_train.shape
        X_train = np.hstack([X_train,np.ones((m,1))])
        y_train = y[train_idx]

        parameters = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        X_val = X[val_idx]
        m,n = X_val.shape
        X_val = np.hstack([X_val,np.ones((m,1))])
        prediction = X_val @ parameters
        predictions[val_idx] = prediction
    
    return predictions.tolist()
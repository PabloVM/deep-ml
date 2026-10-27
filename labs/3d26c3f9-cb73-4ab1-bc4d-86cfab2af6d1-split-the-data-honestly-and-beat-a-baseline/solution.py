# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    idx = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(idx)
    
    n ,m= X.shape
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    
    y_train = y_train.reshape(-1,1)
    
    X_train_mean = np.mean(X_train,axis=0)
    X_train_std = np.std(X_train,axis=0)
    X_train_std = np.where(X_train_std==0,1,X_train_std)
    X_train = (X_train - X_train_mean) / X_train_std
    X_train = np.hstack([X_train,np.ones((X_train.shape[0], 1))])
    
    iterations = 1000
    lr = 0.01
    
    theta = np.zeros((m+1, 1))
    
    
    for step in range(iterations):
        gradient = 1/X_train.shape[0] * X_train.T @ ( X_train @ theta - y_train)
        theta -= gradient * lr
        
    X_test_mean = np.mean(X_test,axis=0)
    X_test_std = np.std(X_test,axis=0)
    X_test_std = np.where(X_test_std==0,1,X_test_std)
    X_test = (X_test - X_test_mean) / X_test_std
    X_test = np.hstack([X_test,np.ones((X_test.shape[0], 1))])
    
    test_predictions = X_test @ theta
    
    return test_predictions.ravel(),train_idx, val_idx,test_idx
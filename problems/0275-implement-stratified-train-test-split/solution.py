import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    class_idx = [np.random.permutation(np.where(y==c)[0]) for c in np.unique(y)]
    class_len = [int(len(c) * test_size) for c in class_idx]

    test_idx = []
    train_idx =[]

    for idx,l in zip(class_idx,class_len):
        
        test_idx.append(idx[:l])
        train_idx.append(idx[l:])
        

    test_idx = np.concatenate(test_idx, axis=0)
    train_idx = np.concatenate(train_idx, axis=0)

    X_test = X[test_idx]
    y_test = y[test_idx]
    X_train = X[train_idx]
    y_train = y[train_idx]

   


    return X_train, X_test, y_train, y_test

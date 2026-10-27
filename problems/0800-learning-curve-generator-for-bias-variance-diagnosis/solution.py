import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    def build_design_matrix(X,degree):
        X = np.asarray(X,dtype=float)
        design_matrix = np.ones((X.shape[0],degree+1))
        for d in range(degree+1):
            design_matrix[:,d] = X[:,0]**d
        return design_matrix

    val_design = build_design_matrix(X_val,degree)

    train_errors = []
    val_errors = []

    for n in train_sizes:
        X = X_train[:n]
        y = y_train[:n]
        train_design = build_design_matrix(X,degree)
        w = np.linalg.pinv(train_design) @ y
        prediction = train_design @ w
        train_mse = np.mean((prediction - y)**2)
        train_errors.append(train_mse)
        prediction = val_design @ w
        val_mse = np.mean((prediction - y_val)**2)
        val_errors.append(val_mse)
    
    if train_errors[-1] > bias_threshold:
        diagnosis = "high_bias"
    elif val_errors[-1] - train_errors[-1] > variance_threshold:
        diagnosis = "high_variance"
    else:
        diagnosis="good_fit"
    
    return{
        "train_errors" : train_errors,
        "val_errors" : val_errors,
        "diagnosis" :diagnosis
    }

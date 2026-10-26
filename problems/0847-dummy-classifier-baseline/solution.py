import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    y_train = np.array(y_train)
    match strategy:
        case "most_frequent":
            most_frequent = np.bincount(y_train).argmax()
            result = [most_frequent for _ in range(n_test)]
        case "constant":
            result = [constant for _ in range(n_test)]
        case "uniform":
            unique  = len(np.unique(y_train))
            sorted_classes = np.sort(y_train)
            result = [sorted_classes[i % unique] for i in range(n_test)]
        case "stratified":
            classes, counts = np.unique(y_train, return_counts=True)
            quotas = n_test * counts / len(y_train)       
            n_per_class = np.floor(quotas).astype(int)    

            remaining = n_test - n_per_class.sum()       
            order = np.argsort(-(quotas - n_per_class), kind="stable")
            n_per_class[order[:remaining]] += 1          

            result = np.repeat(classes, n_per_class).tolist()
        case _:
            raise ValueError("Strategy not valid")
    
    return result

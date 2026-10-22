import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = data.shape[0]
    shuffled_data =np.random.default_rng(seed).permutation(data)
    train_end = int(train_frac * n)
    val_end = int(validation_frac * n)


    train = shuffled_data[:train_end]
    val = shuffled_data[train_end:train_end + val_end]
    test = shuffled_data[train_end+val_end:]

    return [train,val,test]



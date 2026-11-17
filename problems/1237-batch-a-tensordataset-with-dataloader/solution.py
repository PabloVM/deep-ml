import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X,y)
    dataloader = DataLoader(dataset,batch_size=4,shuffle=False)
    num_bathes = 0
    shapes = []
    for batch_x,batch_y in dataloader:
        num_bathes += 1
        shapes.append(batch_x.shape)
    return num_bathes,tuple(shapes[0])


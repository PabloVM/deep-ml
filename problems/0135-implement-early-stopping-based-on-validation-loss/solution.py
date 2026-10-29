from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = 10000
    best_epoch = 0
    epochs_without_improvement = 0
    stop_epoch = len(val_losses) -1
    for i in range(len(val_losses)):
        if best_loss - val_losses[i] > min_delta:
            best_loss = val_losses[i]
            best_epoch = i
            epochs_without_improvement = 0
        else:
            epochs_without_improvement +=1
        
        if epochs_without_improvement >= patience:
            stop_epoch = i
            break
    
    return stop_epoch,best_epoch


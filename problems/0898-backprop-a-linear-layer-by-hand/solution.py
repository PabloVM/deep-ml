import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    grad_input = grad_output @ W 
    grad_W = grad_output.T @ x
    grad_b = torch.sum(grad_output,axis=0)
    return grad_input,grad_W,grad_b

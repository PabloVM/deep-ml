import torch


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save x; return x * sigmoid(x)
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: recover x from ctx, recompute the sigmoid, return grad_output times the local derivative
        (x,)=ctx.saved_tensors
        return grad_output * (torch.sigmoid(x) * (1 + x* (1 - torch.sigmoid(x))))


def swish(x):
    # TODO: SwishFunction.apply(x)
    return SwishFunction.apply(x)

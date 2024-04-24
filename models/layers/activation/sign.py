
import torch


class Sign(torch.autograd.Function):
    """ Sign Activation Function

    Allows for backpropagation of the sign function because it is not differentiable
    """

    @staticmethod
    def forward(ctx, tensor_input):
        ctx.save_for_backward(tensor_input)
        return tensor_input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output * (i.abs() < 1).float()

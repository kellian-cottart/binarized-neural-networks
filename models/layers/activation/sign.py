
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
        grad_i = grad_output.clone()
        grad_i[i.abs() > 1.0] = 0
        return grad_i

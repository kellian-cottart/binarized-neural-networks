
import torch


class Sign(torch.autograd.Function):
    """ Sign Activation Function

    Allows for backpropagation of the sign function because it is not differentiable.
    Uses the hardtanh function to do the backward pass because of the clamping of the gradient.
    """

    @staticmethod
    def forward(ctx, tensor_input, width = 1.0):
        """ Forward pass: sign(input) function"""
        ctx.save_for_backward(tensor_input)
        ctx.width = width
        return tensor_input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass: hardtanh(input) function"""
        i, = ctx.saved_tensors
        width = ctx.width
        return grad_output * (torch.abs(i) < width).float(), None


class SignWeights(torch.autograd.Function):
    """ Sign Binary Weights

    Allows for backpropagation of the binary weights using the identity function.
    This time, the gradient should not be clamped.
    """

    @staticmethod
    def forward(ctx, tensor_input):
        """ Forward pass: sign(input) function"""
        # Returns the sign of the input
        ctx.save_for_backward(tensor_input)
        return tensor_input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass: identity function"""
        # Returns the gradient of input
        return grad_output


class SignActivation(torch.nn.Module):
    """ Sign Activation Layer

    Applies the sign activation function to the input tensor.
    """

    def __init__(self, width=1):
        self.width = width
        super().__init__()

    def forward(self, tensor_input):
        """ Forward pass: sign(input) function"""
        return Sign.apply(tensor_input, self.width)
    
    def extra_repr(self):
        return f"width={self.width}"

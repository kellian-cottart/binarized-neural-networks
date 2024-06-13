
import torch


class Sign(torch.autograd.Function):
    """ Sign Activation Function

    Allows for backpropagation of the sign function because it is not differentiable.
    Uses the hardtanh function to do the backward pass because of the clamping of the gradient.
    """

    @staticmethod
    def forward(ctx, tensor_input):
        """ Forward pass: sign(input) function"""
        ctx.save_for_backward(tensor_input)
        return tensor_input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass: hardtanh(input) function"""
        i, = ctx.saved_tensors
        return grad_output * (torch.abs(i) < 1).float()


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

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return super().__call__(*args, **kwds)

    def forward(self, tensor_input):
        """ Forward pass: sign(input"""
        return Sign.apply(tensor_input)

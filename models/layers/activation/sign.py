from torch.autograd import Function
from torch import sign, abs
from torch.nn import Module


class Sign(Function):
    """ Sign Activation Function

    Allows for backpropagation of the sign function because it is not differentiable.
    Uses the hardtanh function to do the backward pass because of the clamping of the gradient.
    """

    @staticmethod
    def forward(ctx, tensor_input, width=1, offset_x=0.0):
        """ Forward pass: sign(input) function"""
        ctx.save_for_backward(tensor_input)
        ctx.offset_x = offset_x
        ctx.width = width
        return sign(tensor_input - offset_x)

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass: hardtanh(input) function"""
        i, = ctx.saved_tensors
        offset_x = ctx.offset_x
        width = ctx.width
        # condition = ((i >= offset_x - width) & (i <= offset_x + width)).float()
        return grad_output * (abs(i) < width).float(), None, None


class SignWeights(Function):
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


class SignActivation(Module):
    """ Sign Activation Layer

    Applies the sign activation function to the input tensor.
    """

    def __init__(self, width=1, offset_x=0.0):
        """ Initializes the Sign Activation Layer

        Parameters:
            width: width of the hardtanh function used in the backward pass
            offset_x: offset of the sign function
        """
        self.width = width
        self.offset_x = offset_x
        super().__init__()

    def forward(self, tensor_input):
        """ Forward pass: sign(input) function"""
        return Sign.apply(tensor_input, self.width, self.offset_x)

    def extra_repr(self):
        return f"width={self.width}, offset_x={self.offset_x}"

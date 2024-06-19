
import torch


class Gate(torch.autograd.Function):
    """ Sign Elephant Activation Function

    Allows for backpropagation of the gate function because it is not differentiable.
    Backward pass is a soft version of the gate function.
    """

    @staticmethod
    def forward(ctx, tensor_input, width):
        """ Forward pass: gate function: 1 if -1 < input < 1, -1 otherwise"""
        ctx.save_for_backward(tensor_input)
        ctx.width = width
        return 2*(tensor_input.abs() < width).float()-1

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass:
        grad_output * {1 if -w < input < -w/2
                       -1 if w/2 < input < w
                        0 otherwise}
        """
        i, = ctx.saved_tensors
        width = ctx.width
        return grad_output * (((i > -3*width/2) & (i < -width/2)).float() -
                              ((i > width/2) & ((i < 3*width/2))).float()), None


class GateActivation(torch.nn.Module):
    """ Gate Activation Layer

    Applies a gate function to the input tensor.
    """

    def __init__(self, width=1):
        self.width = width
        super(GateActivation, self).__init__()

    def forward(self, tensor_input):
        """ Forward pass: Sign of Elephant function center in 0 with lenght width"""
        return Gate.apply(tensor_input, self.width)

    def extra_repr(self):
        return f"width={self.width}"

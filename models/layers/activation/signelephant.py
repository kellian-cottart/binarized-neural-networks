
import torch


class SignElephant(torch.autograd.Function):
    """ Sign Elephant Activation Function

    Allows for backpropagation of the sign function because it is not differentiable.
    Uses the hardtanh function to do the backward pass because of the clamping of the gradient.
    """

    @staticmethod
    def forward(ctx, tensor_input):
        """ Forward pass: gate function: 1 if -1 < input < 1, 0 otherwise"""
        ctx.save_for_backward(tensor_input)
        return (tensor_input.abs() < 1).float()

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass:
        grad_output * {1 if -1 < input < -0.5
                       -1 if 0.5 < input < 1
                        0 otherwise}
        """
        i, = ctx.saved_tensors
        return grad_output * (((i > -1) & (i < -0.5)).float() -
                              ((i > 0.5) & ((i < 1))).float())


class SignElephantActivation(torch.nn.Module):
    """ Sign Elephant Activation Layer

    Applies the sign elephant activation function to the input tensor.
    """

    def forward(self, tensor_input):
        """ Forward pass: Sign of Elephant function center in 0 with lenght width"""
        return SignElephant.apply(tensor_input)

from torch.nn import Module


class SquaredActivation(Module):
    """ Squared Activation Layer

    Applies the squared activation function to the input tensor.
    """

    def __call__(self, *args: any, **kwds: any):
        return super().__call__(*args, **kwds)

    def forward(self, tensor_input):
        """ Forward pass: input ^ 2"""
        return tensor_input**2

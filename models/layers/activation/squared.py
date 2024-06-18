

import torch


class SquaredActivation(torch.nn.Module):
    """ Squared Activation Layer

    Applies the squared activation function to the input tensor.
    """

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return super().__call__(*args, **kwds)

    def forward(self, tensor_input):
        """ Forward pass: input ^ 2"""
        return tensor_input**2

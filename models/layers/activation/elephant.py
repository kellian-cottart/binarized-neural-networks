
import torch


class ElephantActivation(torch.nn.Module):
    """ Elephant Activation Layer

    Applies the elephant activation function to the input tensor.
    """

    def forward(self, tensor_input):
        """ Forward pass: Elephant function center in 0 with lenght width"""
        return 1/(1+tensor_input**8)


import torch


class GaussianActivation(torch.nn.Module):
    """ Gaussian Activation Layer

    Applies a Gaussian (0, 1) function to the input tensor.
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
        super(GaussianActivation, self).__init__()

    def forward(self, tensor_input):
        """ Forward pass: Gaussian function"""
        return torch.exp(-(tensor_input - self.mean)**2 / (2 * self.std**2))

    def __repr__(self):
        return f"GaussianActivation(mean={self.mean}, std={self.std})"

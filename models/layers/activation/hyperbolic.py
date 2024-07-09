
import torch


class HyperbolicCosineActivation(torch.nn.Module):
    """ Inverse Squared Hyperbolic Cosine Activation Layer
    """

    def __init__(self, width=1):
        self.width = width
        super(HyperbolicCosineActivation, self).__init__()

    def forward(self, tensor_input):
        """ Forward pass: Inverse Squared Hyperbolic Cosine function"""
        return torch.cosh(1/self.width * tensor_input)**(-2)

    def __repr__(self):
        return f"HyperbolicCosineActivation(width={self.width})"

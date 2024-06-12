import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from .activation import *


class BiBayesianLinearConv(torch.nn.Module):
    """ Binary Bayesian Linear Layer using the Gumbel-softmax trick

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        lambda_init (float): Initial value of the lambda parameter
        bias (bool): Whether to use a bias term
        device (torch.device): Device to use for the layer
        dtype (torch.dtype): Data type to use for the layer
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 tau: float = 1.0,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias=False,
                 binarized: bool = False,
                 device: None = None,
                 dtype: None = None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BiBayesianLinearConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.binarized = binarized
        self.tau = tau
        self.weight = nn.Parameter(torch.zeros(
            out_channels, in_channels, kernel_size, kernel_size, **factory_kwargs))

    def sample(self, x, n_samples=1):
        """ Sample the weights for the layer"""

    def forward(self, x, n_samples=1):
        """ Forward pass of the neural network for the backward pass """
        # Compute epsilon from uniform U(0,1), but avoid 0
        epsilon = torch.distributions.Uniform(
            1e-10, 1).sample((n_samples, *self.weight.shape)).to(x.device)
        # Compute delta = 1/2 log(epsilon/(1-epsilon))
        delta = 0.5 * torch.log(epsilon/(1-epsilon))

        relaxed_weights = torch.tanh(self.weight + delta)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, lambda.shape={}'.format(
            self.in_features, self.out_features, self.weight.shape)

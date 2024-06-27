import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from .activation import *


class BiBayesianConv(torch.nn.Module):
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
                 in_features: int,
                 out_features: int,
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
        super(BiBayesianConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.binarized = binarized
        self.tau = tau
        self.weight = nn.Parameter(torch.zeros(
            out_features, in_features, kernel_size, kernel_size, **factory_kwargs))

    def sample(self, x, n_samples=1):
        """ Sample the weights for the layer and do the forward pass"""
        # Compute p for Bernoulli sampling
        p = torch.sigmoid(2*self.weight)
        # Sample the weights according to 2*Ber(p) - 1
        weights = 2*Bernoulli(p).sample((n_samples,)).to(x.device)-1
        if x.dim() == 4:
            x = x.unsqueeze(0)
        # weights: sfckl (samples, filters, channels, kernel_height (k), kernel_width (l))
        # x: sbcwh (samples, batch, channels, width, height)
        output = torch.einsum('sfckl, sbcwh -> sbfwh', weights, x)
        return output

    def forward(self, x, n_samples=1):
        """ Forward pass of the neural network for the backward pass """
        # Compute epsilon from uniform U(0,1), but avoid 0
        epsilon = torch.distributions.Uniform(
            1e-10, 1).sample(sample_shape=(n_samples, *self.weight.shape)).to(x.device)
        # Compute delta = 1/2 log(epsilon/(1-epsilon))
        delta = 0.5 * torch.log(epsilon/(1-epsilon))
        # Compute the new relaxed weights values
        relaxed_weights = torch.tanh((1/self.tau) * (self.weight + delta))
        # Compute the output of the layer
        if x.dim() == 4:
            x = x.unsqueeze(0)
        # relaxed_weights: sfckl (samples, filters, channels, kernel_height (k), kernel_width (l))
        # x: sbcwh (samples, batch, channels, width, height)
        # We want the following output: sbfwh (samples, batch, filters, width, height)
        output = torch.einsum('sfckl, sbcwh -> sbfwh', relaxed_weights, x)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias)

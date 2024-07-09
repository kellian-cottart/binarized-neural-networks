import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from .activation import *


class BiBayesianLinear(torch.nn.Module):
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
                 binarized: bool = False,
                 device: None = None,
                 dtype: None = None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BiBayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features,
                                               in_features,
                                               **factory_kwargs))
        self.binarized = binarized
        self.tau = tau

    def sample(self, x, n_samples=1):
        """ Sample the weights for the layer"""
        # Compute p for Bernoulli sampling
        p = torch.sigmoid(2*self.weight)
        # Sample the weights according to 2*Ber(p) - 1
        weights = 2*Bernoulli(p).sample((n_samples,)).to(x.device)-1
        # Notation: s samples, b batch, o out_features, i in_features
        return torch.einsum('soi, sbi -> sbo', weights, x)

    def forward(self, x, n_samples=1):
        """ Forward pass of the neural network for the backward pass """
        # Compute epsilon from uniform U(0,1), but avoid 0
        epsilon = torch.distributions.Uniform(
            1e-10, 1).sample((n_samples, *self.weight.shape)).to(x.device)
        # Compute delta = 1/2 log(epsilon/(1-epsilon))
        delta = (0.5 * torch.log(epsilon/(1-epsilon))).to(x.device)
        # Compute the new relaxed weights values
        if self.binarized:
            relaxed_weights = SignWeights.apply(
                (self.weight + delta))
        else:
            relaxed_weights = torch.tanh(
                (self.weight + delta)/self.tau)
        # just a little bit faster if we have one sample
        if relaxed_weights.shape[0] == 1:
            return (x.squeeze(0) @ relaxed_weights.squeeze(0).T).unsqueeze(0)
        return torch.einsum('soi, sbi -> sbo', relaxed_weights, x)

    def extra_repr(self):
        return 'in_features={}, out_features={}, lambda.shape={}'.format(
            self.in_features, self.out_features, self.weight.shape)

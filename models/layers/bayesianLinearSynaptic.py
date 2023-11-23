# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:02:27 2023

@author: DB262466
"""


import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

__all__ = ['MetaBayesLinearParallel']


class GaussianWeightParallel(object):
    """Represents the Gaussian distribution for weights in the Bayesian neural network."""

    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu  # Mean of the distribution
        self.sigma = sigma  # Standard deviation of the distribution
        self.normal = torch.distributions.Normal(
            0, 1)  # Standard normal distribution

    def sample(self, samples=1):
        """Sample from the Gaussian distribution using the reparameterization trick."""
        if samples == 0:
            # Use the mean value for inference
            return torch.stack((self.mu.T,) * 1)
        else:
            # Sample from the standard normal and adjust with sigma and mu
            epsilon = self.normal.sample(
                (samples, self.sigma.size()[1], self.sigma.size()[0])).to(self.mu.device)
        return torch.stack((self.mu.T,) * samples) + torch.stack((self.sigma.T,) * samples) * epsilon


class GaussianBiasParallel(object):
    """Represents the Gaussian distribution for biases in the Bayesian neural network."""

    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.normal = torch.distributions.Normal(0, 1)

    def sample(self, samples=1):
        """Sample from the Gaussian distribution using the reparameterization trick."""
        if samples == 0:
            return torch.stack((self.mu,) * 1)
        else:
            epsilon = self.normal.sample(
                (samples, self.sigma.size()[0])).to(self.mu.device)
        return torch.stack((self.mu,) * samples) + torch.stack((self.sigma,) * samples) * epsilon


class MetaBayesLinearParallel(Module):
    """Bayesian linear layer using parallelized Gaussian distributions for weights and biases."""
    __constants__ = ['in_features', 'out_features',
                     'SNR', 'MaxMean', 'MinSigma']

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.01,
                 bias: bool = True, zeroMean: bool = False, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MetaBayesLinearParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameters
        self.weight_sigma = Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.weight_mu = Parameter(torch.empty(
            (out_features, in_features), **factory_kwargs))
        self.sigma0 = 1 / math.sqrt(in_features)
        self.weight = GaussianWeightParallel(self.weight_mu, self.weight_sigma)

        # Control for zero mean initialization
        self.zeroMean = zeroMean
        self.sigma_init = sigma_init

        # Initialize bias if applicable
        if bias:
            self.bias_sigma = Parameter(
                torch.empty(out_features, **factory_kwargs))
            self.bias_mu = Parameter(torch.empty(
                out_features, **factory_kwargs))
            self.bias = GaussianBiasParallel(self.bias_mu, self.bias_sigma)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        if not self.zeroMean:
            init.uniform_(self.weight_mu, -self.sigma0, self.sigma0)
        if self.zeroMean:
            init.constant_(self.weight_mu, 0)
        init.constant_(self.weight_sigma, self.sigma_init)

        if self.bias is not None:
            init.uniform_(self.bias_mu, -self.sigma0, self.sigma0)
            init.constant_(self.bias_sigma, self.sigma_init)

    def forward(self, input: Tensor, samples: int) -> Tensor:
        """Forward pass using sampled weights and biases."""
        W = self.weight.sample(samples)
        if self.bias:
            B = self.bias.sample(samples)
            return torch.matmul(input, W) + B[:, None]
        else:
            return torch.matmul(input, W)

    def extra_repr(self) -> str:
        """Representation for pretty print and debugging."""
        return 'in_features={}, out_features={}, sigma_init={}, bias={}'.format(
            self.in_features, self.out_features, self.sigma_init, self.bias is not None)

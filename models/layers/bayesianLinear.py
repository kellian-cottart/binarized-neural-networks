from math import sqrt
from torch import Tensor
import torch.nn.init as init
from torch.nn.modules import Module
from .gaussianParameter import *
from torch import einsum


class MetaBayesLinearParallel(Module):
    """Bayesian linear layer using parallelized Gaussian distributions for weights and biases."""
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, zeroMean: bool = False, sigma_init=0.1, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MetaBayesLinearParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameters
        self.bound = sqrt(2/in_features)
        self.sigma_init = sigma_init
        self.weight = GaussianParameter(out_features=out_features,
                                        in_features=in_features,
                                        **factory_kwargs)
        # Control for zero mean initialization
        self.zeroMean = zeroMean

        # Initialize bias if applicable
        if bias:
            self.bias = GaussianParameter(out_features=out_features,
                                          **factory_kwargs)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        if self.zeroMean:
            # put all the mean value to zero
            init.uniform_(self.weight.mu, 0)
        else:
            # like kaiming uniform initialisation
            init.uniform_(self.weight.mu, -self.bound, self.bound)
        init.constant_(self.weight.sigma, self.sigma_init)
        if self.bias is not None:
            # bias mean value always intialize to zero
            init.constant_(self.bias.mu, 0)
            init.constant_(self.bias.sigma, self.sigma_init)

    def forward(self, x: Tensor, samples: int) -> Tensor:
        """Forward pass using sampled weights and biases."""
        W = self.weight.sample(samples)
        x = x.reshape(samples, x.size(0)//samples, x.size(1))
        if self.bias:
            B = self.bias.sample(samples).unsqueeze(1).repeat(1, x.size(1), 1)
            out = einsum('soi, sbi -> sbo', W, x) + B
        else:
            out = einsum('soi, sbi -> sbo', W, x)
        return out.reshape(out.size(0)*out.size(1), *out.size()[2:])

    def extra_repr(self) -> str:
        """Representation for pretty print and debugging."""
        return 'in_features={}, out_features={}, sigma_init={}, bias={}'.format(
            self.in_features, self.out_features, self.sigma_init, self.bias is not None)

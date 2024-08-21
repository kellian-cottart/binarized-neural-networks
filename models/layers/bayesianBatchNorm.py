import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
from .gaussianParameter import *


class MetaBayesBatchNormNd(Module):
    """As in pytorch, except the weights are gaussian."""

    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'affine', 'weight', 'bias']
    __annotations__ = {'running_mean': Optional[Tensor],
                       'running_var': Optional[Tensor],
                       'num_batches_tracked': Optional[Tensor],
                       'weight': Optional[Tensor],
                       'bias': Optional[Tensor]}
    track_running_stats: bool
    momentum: float
    eps: float
    affine: bool
    weight: Optional[Tensor]
    bias: Optional[Tensor]
    running_mean: Optional[Tensor]
    running_var: Optional[Tensor]
    num_features: int
    samples: int
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MetaBayesBatchNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.dtype = dtype

        if self.affine:
            self.weight_mu = Parameter(
                torch.empty(num_features, **factory_kwargs))
            self.weight_sigma = Parameter(
                torch.empty(num_features, **factory_kwargs))
            self.weight = GaussianParameter(self.weight_mu, self.weight_sigma)
            self.bias_mu = Parameter(torch.empty(
                num_features, **factory_kwargs))
            self.bias_sigma = Parameter(
                torch.empty(num_features, **factory_kwargs))
            self.bias = GaussianParameter(self.bias_mu, self.bias_sigma)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(
                num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(
                num_features, **factory_kwargs))
            self.register_buffer('num_batches_tracked', torch.tensor(
                0, dtype=torch.long, **factory_kwargs))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.constant_(self.weight_mu, 1)
            init.constant_(self.weight_sigma, 0.1)
            init.constant_(self.bias_mu, 0)
            init.constant_(self.bias_sigma, 0.1)
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaBayesBatchNorm2d(MetaBayesBatchNormNd):
    """
    As in pytorch, except the weights are gaussian.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, device=None, dtype=None) -> None:
        super(MetaBayesBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)

    def forward(self, x: Tensor, samples=1):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            x = x.repeat(samples, 1, 1, 1, 1)
        x = x.view(x.size(1), x.size(0)*x.size(2), x.size(3), x.size(4))
        if self.affine:
            W = self.weight.sample(samples)
            W = W.view(W.size(0)*W.size(1), 1, 1, 1)
            B = self.bias.sample(samples).flatten()
            B = B.view(B.size(0), 1, 1, 1)
            out = F.batch_norm(x, self.running_mean, self.running_var, W, B, self.training,
                               self.momentum, self.eps)
        else:
            out = F.batch_norm(x, self.running_mean, self.running_var, None, None, self.training,
                               self.momentum, self.eps)
        return out.view(samples, out.size(0), out.size(1) // samples, out.size(2), out.size(3))


class MetaBayesBatchNorm1d(MetaBayesBatchNormNd):
    """
    As in pytorch, except the weights are gaussian.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, device=None, dtype=None) -> None:
        super(MetaBayesBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)

    def forward(self, x: Tensor, samples=1):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            x = x.repeat(samples, 1, 1)
        x = x.view(x.size(1), x.size(0)*x.size(2))
        if self.affine:
            W = self.weight.sample(samples)
            W = W.view(W.size(0)*W.size(1), 1, 1)
            B = self.bias.sample(samples).flatten()
            B = B.view(B.size(0), 1, 1)
            out = F.batch_norm(x, self.running_mean, self.running_var, W, B, self.training,
                               self.momentum, self.eps)
        else:
            out = F.batch_norm(x, self.running_mean, self.running_var, None, None, self.training,
                               self.momentum, self.eps)
        return out.view(samples, out.size(0), out.size(1) // samples)

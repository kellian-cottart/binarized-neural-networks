from torch.nn.modules.pooling import _AdaptiveAvgPoolNd
from torch.nn.common_types import _size_2_opt_t
from torch.nn import Sequential
from torch import Tensor
from torch.nn import functional as F


class MetaBayesianAdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes depending on the number of samples.
    """

    output_size: _size_2_opt_t

    def forward(self, x: Tensor, samples: int = 5) -> Tensor:
        if x.size(0) != samples:
            raise ValueError(
                "Input tensor must have the same number of samples as the number of samples in the forward pass.")
        else:
            x_reshaped = x.reshape([x.shape[0]*x.shape[1], *x.shape[2:]])
            out = F.adaptive_avg_pool2d(x_reshaped, self.output_size)
            out = out.reshape([x.shape[0], x.shape[1], *out.shape[1:]])
            return out


class MetaBayesSequential(Sequential):
    """Sequential container for Bayesian layers"""

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, samples=5):
        for module in self:
            try:
                x = module(x, samples)
            except:
                x = module(x)
        return x

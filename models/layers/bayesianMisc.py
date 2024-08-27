from torch.nn import Sequential, Module
from torchvision.models.efficientnet import MBConv, _MBConvConfig
from typing import Callable, Optional
from torch.nn import Flatten, Module, AdaptiveAvgPool2d, Sigmoid, ReLU
from torch import Tensor, no_grad
from .bayesianConv import MetaBayesConv2d


class MetaBayesSequential(Sequential):
    """Sequential container for Bayesian layers"""

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, samples):
        for module in self:
            if "Meta" in module.__class__.__name__:
                x = module(x, samples)
            else:
                x = module(x)
        return x


class MetaBayesMBConv(MBConv):
    """ MBConv block with different forward pass for Bayesian layers
    https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py"""

    def forward(self, input, samples: int):
        result = self.block(input, samples)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class MetaBayesSqueezeExcitation(Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., Module], optional): ``delta`` activation. Default: ``ReLU``
        scale_activation (Callable[..., Module]): ``sigma`` activation. Default: ``Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., Module] = ReLU,
        scale_activation: Callable[...,
                                   Module] = Sigmoid,
        sigma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc1 = MetaBayesConv2d(
            input_channels, squeeze_channels, 1, sigma_init=sigma_init)
        self.fc2 = MetaBayesConv2d(
            squeeze_channels, input_channels, 1, sigma_init=sigma_init)
        self.activation = activation
        self.scale_activation = scale_activation

    def set_weights(self, fc1, fc2):
        self.fc1.weight.mu.data = fc1.weight.data.clone()
        self.fc2.weight.mu.data = fc2.weight.data.clone()
        if fc1.bias is not None and fc2.bias is not None:
            self.fc1.bias.mu.data = fc1.bias.data.clone()
            self.fc2.bias.mu.data = fc2.bias.data.clone()

    def forward(self, x: Tensor, samples: int) -> Tensor:
        scale = MetaBayesSequential(
            self.avgpool,
            self.fc1,
            self.activation,
            self.fc2,
            self.scale_activation,
        )(x, samples)
        return x * scale

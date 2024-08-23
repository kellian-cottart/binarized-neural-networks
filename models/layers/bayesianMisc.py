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
        for i, module in enumerate(self):
            if "Meta" in module.__class__.__name__:
                x = module(x, samples)
            elif not isinstance(module, Flatten):
                cat_x = x.reshape(x.size(0)*x.size(1), *x.size()[2:])
                cat_x = module(cat_x)
                x = cat_x.reshape(x.size(0), x.size(1), *cat_x.size()[1:])
            else:
                x = module(x)
        return x


class MetaBayesMBConv(MBConv):
    """ MBConv block with different forward pass for Bayesian layers
    https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py"""

    def forward(self, x, samples: int):
        result = self.block(x, samples)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += x
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
    ) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc1 = MetaBayesConv2d(input_channels, squeeze_channels, 1)
        self.fc2 = MetaBayesConv2d(squeeze_channels, input_channels, 1)
        self.activation = activation
        self.scale_activation = scale_activation

    def set_weights(self, fc1, fc2):
        with no_grad():
            self.fc1.weight.mu.copy_(fc1.weight)
            self.fc2.weight.mu.copy_(fc2.weight)
            if fc1.bias is not None and fc2.bias is not None:
                self.fc1.bias.mu.copy_(fc1.bias)
                self.fc2.bias.mu.copy_(fc2.bias)

    def forward(self, x: Tensor, samples: int) -> Tensor:
        scale = MetaBayesSequential(
            self.avgpool,
            self.fc1,
            self.activation,
            self.fc2,
            self.scale_activation,
        )(x, samples)
        return x * scale

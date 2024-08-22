from torch.nn import Sequential, Module
from torchvision.models.efficientnet import MBConv, _MBConvConfig
from typing import Callable, Optional


class MetaBayesSequential(Sequential):
    """Sequential container for Bayesian layers"""

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, samples):
        for module in self:
            try:
                x = module(x, samples)
            except:
                try:
                    x = module(x)
                except:
                    cat_x = x.reshape(x.size(0)*x.size(1), *x.size()[2:])
                    cat_x = module(cat_x)
                    x = cat_x.reshape(x.size(0), x.size(1), *cat_x.size()[1:])
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

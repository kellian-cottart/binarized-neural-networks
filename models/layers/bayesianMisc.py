from torch.nn import Sequential
from torchvision.models.efficientnet import MBConv


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
                    cat_x = x.view([x.shape[0]*x.shape[1], *x.shape[2:]])
                    cat_x = module(cat_x)
                    x = cat_x.view([x.shape[0], x.shape[1], *cat_x.shape[1:]])
        return x


class MetaBayesMBConv(MBConv):
    """ MBConv block with different forward pass for Bayesian layers
    https: // github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py"""

    def forward(self, x, samples: int):
        result = self.block(x, samples)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

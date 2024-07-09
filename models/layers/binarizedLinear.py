import torch
from .activation.sign import Sign, SignWeights


class BinarizedLinear(torch.nn.Linear):
    """ Binarized Linear Layer

    Args:
        latent_weights (bool): Whether to use latent weights or not
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=False,
                 device='cuda:0'
                 ):
        super(BinarizedLinear, self).__init__(
            in_features, out_features, bias=bias, device=device)

    def forward(self, input):
        """Forward propagation of the binarized linear layer"""
        return torch.nn.functional.linear(input, SignWeights.apply(self.weight), None if not self.bias else SignWeights.apply(self.bias))

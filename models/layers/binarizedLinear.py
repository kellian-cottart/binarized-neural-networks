import torch
from .activation.sign import Sign


class BinarizedLinear(torch.nn.Linear):
    """ Binarized Linear Layer

    Args:
        latent_weights (bool): Whether to use latent weights or not
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=False,
                 device='cuda'
                 ):
        super(BinarizedLinear, self).__init__(
            in_features, out_features, bias=bias, device=device)

    @torch.jit.export
    def forward(self, input):
        """Forward propagation of the binarized linear layer"""
        return torch.nn.functional.linear(input, Sign.apply(self.weight))

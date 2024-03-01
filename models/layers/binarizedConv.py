
import torch
from .activation.sign import Sign


class BinarizedConv2d(torch.nn.Conv2d):
    """ Binarized Convolutional Linear Layer

    Args:
        latent_weights (bool): Whether to use latent weights or not
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int = 1,
                 bias=False,
                 latent_weights=False,
                 device='cuda'
                 ):
        super(BinarizedConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            device=device)
        self.latent_weights = latent_weights

    def forward(self, input):
        """Forward propagation of the binarized linear layer"""
        if not self.latent_weights:
            self.weight.data = self.weight.data.sign()
            if self.bias is not False and self.bias is not None:
                self.bias.data = self.bias.data.sign()
            return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        else:
            if self.bias is not None:
                return torch.nn.functional.conv2d(input, Sign.apply(self.weight), Sign.apply(self.bias), self.stride, self.padding)
            else:
                return torch.nn.functional.conv2d(input, Sign.apply(self.weight), None, self.stride, self.padding)

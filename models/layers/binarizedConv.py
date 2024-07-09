
import torch
from .activation.sign import SignWeights


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

    def forward(self, input):
        """Forward propagation of the binarized linear layer"""
        if self.bias is not None:
            output = torch.nn.functional.conv2d(input, SignWeights.apply(
                self.weight), SignWeights.apply(self.bias), self.stride, self.padding)
        else:
            output = torch.nn.functional.conv2d(input, SignWeights.apply(
                self.weight), None, self.stride, self.padding)
        return output

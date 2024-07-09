
import torch
from .layers import *
from .convNeuralNetwork import ConvNN
from .binaryNeuralNetwork import *
from typing import Union


class ConvBiNN(ConvNN):
    """ Convolutional Binarized Neural Network (ConvBiNN)
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
                 features: list = [64, 128, 256],
                 init: str = "uniform",
                 std: float = 0.01,
                 device: str = "cuda:0",
                 dropout: bool = False,
                 normalization: str = None,
                 bias: bool = False,
                 running_stats: bool = False,
                 affine: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.15,
                 activation_function: torch.nn.functional = torch.nn.functional.relu,
                 output_function: str = "softmax",
                 kernel_size: int = 3,
                 padding: Union[int, tuple] = "same",
                 stride: int = 1,
                 dilation: int = 1,
                 gnnum_groups: int = 32,
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            layers (list): List of layer sizes for the classifier
            features (list): List of layer sizes for the feature extractor
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
            device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
            dropout (bool): Whether to use dropout
            batchnorm (bool): Whether to use batchnorm
            bias (bool): Whether to use bias
            bneps (float): BatchNorm epsilon
            bnmomentum (float): BatchNorm momentum
            running_stats (bool): Whether to use running stats in BatchNorm
            affine (bool): Whether to use affine transformation in BatchNorm
            activation_function (torch.nn.functional): Activation function
            output_function (str): Output function
        """
        super().__init__(layers=layers, features=features, init=init, std=std, device=device,
                         dropout=dropout, normalization=normalization, bias=bias,
                         running_stats=running_stats, affine=affine, eps=eps, momentum=momentum,
                         activation_function=activation_function, output_function=output_function,
                         kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                         gnnum_groups=gnnum_groups, *args, **kwargs)
        self.classifier = BiNN(layers=layers, init=init, std=std, device=device,
                               dropout=dropout, normalization=normalization, bias=bias,
                               running_stats=running_stats, affine=affine, eps=eps, momentum=momentum,
                               activation_function=activation_function, output_function=output_function)

    def _features_init(self, features, bias=False):
        """ Initialize layers of the network for convolutional layers

            Args:
                features (list): List of layer sizes for the feature extractor
                bias (bool): Whether to use bias or not
        """
        # Add conv layers to the network as well as batchnorm and maxpool
        for i, _ in enumerate(features[:-1]):
            # Conv layers with BatchNorm and MaxPool
            self.features.append(BinarizedConv2d(features[i], features[i+1], kernel_size=self.kernel_size,
                                 padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(BinarizedConv2d(features[i+1], features[i+1], kernel_size=self.kernel_size,
                                 padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(torch.nn.MaxPool2d(kernel_size=2))
            if self.dropout == True:
                self.features.append(torch.nn.Dropout2d(p=0.2))

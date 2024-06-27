
import torch
from .layers import *
from .convNeuralNetwork import ConvNN
from .biBayesianNeuralNetwork import *
from typing import Union


class ConvBiBayesianNeuralNetwork(ConvNN):
    """ Convolutional Binarized Neural Network(ConvBiNN)
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
                 features: list = [64, 128, 256],
                 tau=1.0,
                 n_samples_forward: int = 1,
                 n_samples_backward: int = 1,
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

        self.tau = tau
        self.n_samples_forward = n_samples_forward
        self.n_samples_backward = n_samples_backward
        super().__init__(layers=layers, features=features, init=init, std=std, device=device,
                         dropout=dropout, normalization=normalization, bias=bias,
                         running_stats=running_stats, affine=affine, eps=eps, momentum=momentum,
                         activation_function=activation_function, output_function=output_function,
                         kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                         gnnum_groups=gnnum_groups, *args, **kwargs)
        self.classifier = BiBayesianNN(layers=layers, init=init, std=std, device=device,
                                       dropout=dropout, normalization=normalization, bias=bias,
                                       running_stats=running_stats, affine=affine, eps=eps, momentum=momentum,
                                       activation_function=activation_function, output_function=output_function,
                                       tau=tau, n_samples_forward=n_samples_forward, n_samples_backward=n_samples_backward)

    def _features_init(self, features, bias=False):
        """ Initialize layers of the network for convolutional layers

            Args:
                features(list): List of layer sizes for the feature extractor
                bias(bool): Whether to use bias or not
        """
        # Add conv layers to the network as well as batchnorm and maxpool
        for i, _ in enumerate(features[:-1]):
            # Conv layers with BatchNorm and MaxPool
            self.features.append(BiBayesianConv(features[i], features[i+1], tau=self.tau, kernel_size=self.kernel_size,
                                 padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(BiBayesianConv(features[i+1], features[i+1], tau=self.tau, kernel_size=self.kernel_size,
                                 padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(torch.nn.MaxPool2d(kernel_size=2))
            if self.dropout == True:
                self.features.append(torch.nn.Dropout2d(p=0.2))

    def forward(self, x, backwards=True, *args, **kwargs):
        """Forward propagation of the binarized neural network"""
        for layer in self.features:
            if isinstance(layer, BiBayesianConv):
                if backwards == True:
                    x = layer.forward(x, self.n_samples_backward)
                else:
                    x = layer.sample(x, self.n_samples_forward)
            else:
                try:
                    x = layer(x)
                except:
                    # Normalization layers, but input is (n_samples, batch, features)
                    shape = x.shape
                    x = x.reshape([shape[0]*shape[1], *x.shape[2:]])
                    x = layer(x)
                    x = x.reshape([shape[0], shape[1], *x.shape[1:]])
        return self.classifier.forward(x, backwards=backwards)

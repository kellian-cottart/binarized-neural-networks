import torch
from typing import Union

from models.layers.activation.sign import SignActivation
from .deepNeuralNetwork import DNN
from .layers.activation import *


class ConvNN(torch.nn.Module):
    """ Convolutional Neural Network Base Class
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
                 activation_function: str = "relu",
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
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.device = device
        self.features = torch.nn.ModuleList().to(self.device)
        self.dropout = dropout
        self.normalization = normalization
        self.eps = eps
        self.momentum = momentum
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.gnnum_groups = gnnum_groups
        ### LAYER INITIALIZATION ###
        self._features_init(features, bias)
        ### WEIGHT INITIALIZATION ###
        self._weight_init(init, std)
        self.classifier = DNN(layers, init, std, device, dropout, normalization, bias, running_stats,
                              affine, eps, momentum, gnnum_groups, activation_function)

    def _features_init(self, features, bias=False):
        """ Initialize layers of the network for convolutional layers

            Args:
                features (list): List of layer sizes for the feature extractor
                bias (bool): Whether to use bias or not
        """
        # Add conv layers to the network as well as batchnorm and maxpool
        for i, _ in enumerate(features[:-1]):
            # Conv layers with BatchNorm and MaxPool
            self.features.append(torch.nn.Conv2d(features[i], features[i+1], kernel_size=self.kernel_size,
                                 padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(torch.nn.Conv2d(features[i+1], features[i+1], kernel_size=self.kernel_size,
                                 padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(torch.nn.MaxPool2d(kernel_size=2))
            self.features.append(torch.nn.Dropout2d(p=0.2))

    def _activation_init(self):
        """
        Returns:
            torch.nn.Module: Activation function module
        """
        activation_functions = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "tanh": torch.nn.Tanh,
            "sign": SignActivation,
            "squared": SquaredActivation,
            "elephant": ElephantActivation,
            "gate": GateActivation
        }
        # add parameters to activation function if needed
        try:
            return activation_functions.get(self.activation_function, torch.nn.Identity)(**self.activation_parameters).to(self.device)
        except:
            return activation_functions.get(self.activation_function, torch.nn.Identity)().to(self.device)

    def _norm_init(self, n_features):
        """Returns a layer of normalization"""
        if self.normalization == "batchnorm":
            return torch.nn.BatchNorm2d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats).to(self.device)
        elif self.normalization == "layernorm":
            return torch.nn.LayerNorm(n_features).to(self.device)
        elif self.normalization == "instancenorm":
            return torch.nn.InstanceNorm2d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats).to(self.device)
        elif self.normalization == "groupnorm":
            return torch.nn.GroupNorm(self.gnnum_groups, n_features).to(self.device)
        else:
            return torch.nn.Identity().to(self.device)

    def _weight_init(self, init='normal', std=0.1):
        """ Initialize weights of each layer

        Args:
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
        """

        for layer in self.features:
            if isinstance(layer, torch.nn.Module) and hasattr(layer, 'weight') and layer.weight is not None:
                if init == 'gaussian':
                    torch.nn.init.normal_(
                        layer.weight.data, mean=0.0, std=std)
                elif init == 'uniform':
                    torch.nn.init.uniform_(
                        layer.weight.data, a=-std/2, b=std/2)
                elif init == 'xavier':
                    torch.nn.init.xavier_normal_(layer.weight.data)

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        for layer in self.features:
            x = layer(x)
        return self.classifier.forward(x)

import torch
from torch.nn import *
from typing import Union
from .deepNeuralNetwork import DNN
from .layers.activation import *


class MidVGG(Module):
    """ Convolutional Neural Network Base Class
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
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
        self.device = device
        self.dropout = dropout
        self.normalization = normalization
        self.eps = eps
        self.momentum = momentum
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.gnnum_groups = gnnum_groups
        ### LAYER INITIALIZATION ###
        self.features = self._features_init()
        ### WEIGHT INITIALIZATION ###
        self._weight_init(init, std)
        ## CLASSIFIER INITIALIZATION ##
        self.classifier = DNN(layers, init, std, device, dropout, normalization, bias, running_stats,
                              affine, eps, momentum, gnnum_groups, activation_function)

    def block(self, in_channels, out_channels, num_conv_layers, kernel_size, padding, stride):
        """ Create a block of convolutional layers

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_conv_layers (int): Number of convolutional layers
            kernel_size (int): Kernel size for convolutional layers
            padding (int): Padding for convolutional layers
            stride (int): Stride for convolutional layers

        Returns:
            torch.nn.Module: Block of convolutional layers
        """
        layers = []
        for _ in range(num_conv_layers):
            layers.append(Conv2d(in_channels, out_channels,
                          kernel_size=kernel_size, padding=padding, stride=stride))
            layers.append(self._norm_init(out_channels))
            layers.append(self._activation_init())
            in_channels = out_channels
        layers.append(MaxPool2d(kernel_size=2, stride=2))
        return torch.nn.ModuleList(layers).to(self.device)

    def _features_init(self):
        """ Initialize layers of the network for mid-VGG architecture
        """
        return torch.nn.ModuleList(
            # input size is 3x128x128
            self.block(3, 64, 2, 3, 1, 1) +  # 64x64x64
            self.block(64, 128, 2, 3, 1, 1) +  # 128x32x32
            self.block(128, 256, 3, 3, 1, 1) +  # 256x16x16
            self.block(256, 512, 3, 3, 1, 1) +  # 512x8x8
            self.block(512, 512, 3, 3, 1, 1)  # 512x4x4
        ).to(self.device)

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

    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return super().extra_repr() + f"parameters={self.number_parameters()}"

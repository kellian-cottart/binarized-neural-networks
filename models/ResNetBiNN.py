import torch
from typing import Union
from .layers import *
import torchvision


class ResNetBiNN(torch.nn.Module):
    """ Convolutional Neural Network Base Class
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
                 init: str = "uniform",
                 std: float = 0.01,
                 device: str = "cuda:0",
                 dropout: bool = False,
                 batchnorm: bool = False,
                 bias: bool = False,
                 running_stats: bool = False,
                 affine: bool = False,
                 bneps: float = 1e-5,
                 bnmomentum: float = 0.15,
                 activation_function: torch.nn.functional = torch.nn.functional.relu,
                 output_function: str = "softmax",
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
        super(ResNetBiNN, self).__init__()
        self.device = device
        self.layers = torch.nn.ModuleList().to(self.device)
        self.features = torch.nn.ModuleList().to(self.device)
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.bneps = bneps
        self.bnmomentum = bnmomentum
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.output_function = output_function
        ### LAYER INITIALIZATION ###

        self.features = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT).to(self.device)
        # Remove the last layer of the network
        self.features = torch.nn.Sequential(
            *list(self.features.children())[:-1])
        # Freeze the weights of the feature extractor
        # for param in self.features.parameters():
        #     param.requires_grad = False

        self._classifier_init(layers, bias)
        ### WEIGHT INITIALIZATION ###
        self._weight_init(init, std)

    def _classifier_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            dropout (bool): Whether to use dropout
            bias (bool): Whether to use bias
        """
        # Add the fully connected layers to predict the output
        for i, _ in enumerate(layers[:-1]):
            # Linear layers with BatchNorm
            if self.dropout and i != 0:
                self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(BinarizedLinear(
                layers[i],
                layers[i+1],
                bias=bias,
                device=self.device,
                latent_weights=False))
            if self.batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(
                    layers[i+1],
                    affine=self.affine,
                    track_running_stats=self.running_stats,
                    device=self.device,
                    eps=self.bneps,
                    momentum=self.bnmomentum))

    def _weight_init(self, init='normal', std=0.1):
        """ Initialize weights of each layer

        Args:
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
        """

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                if init == 'gauss':
                    torch.nn.init.normal_(
                        layer.weight, mean=0.0, std=std)
                elif init == 'uniform':
                    torch.nn.init.uniform_(
                        layer.weight, a=-std/2, b=std/2)
                elif init == 'xavier':
                    torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        """ Forward pass of DNN

        Args: 
            x (torch.Tensor): Input tensor

        Returns: 
            torch.Tensor: Output tensor

        """
        ### FORWARD PASS FEATURES ###
        x = self.features.forward(x)
        # Flatten the output of the feature extractor
        x = x.view(x.size(0), -1)
        ### FORWARD PASS CLASSIFIER ###
        unique_layers = set(type(layer) for layer in self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if layer is not self.layers[-1] and (i+1) % len(unique_layers) == 0:
                x = self.activation_function(x)
        if self.output_function == "softmax":
            x = torch.nn.functional.softmax(x, dim=1)
        if self.output_function == "log_softmax":
            x = torch.nn.functional.log_softmax(x, dim=1)
        if self.output_function == "sigmoid":
            x = torch.nn.functional.sigmoid(x)

        return x

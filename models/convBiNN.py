
import torch
from .layers import *
from .convNeuralNetwork import ConvNN


class ConvBiNN(ConvNN):
    """ Convolutional Binarized Neural Network (ConvBiNN)

    Neural Network with binary weights and activations, using hidden weights called "degrees of certainty" (DOCs) to approximate real-valued weights.

    Args:
        layers (list): List of layer sizes for the classifier
        features (list): List of layer sizes for the feature extractor
        init (str): Initialization method for weights
        std (float): Standard deviation for initialization
        device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
        dropout (bool): Whether to use dropout
        latent_weights (bool): Whether to use latent weights or not
    """

    def __init__(self, *args, **kwargs):
        self.latent_weights = kwargs['latent_weights'] if 'latent_weights' in kwargs else True
        super().__init__(*args, **kwargs)

    def _features_init(self, features, bias=False):
        """ Initialize layers of the network for convolutional layers

            Args:
                layers (list): List of layer sizes
                bias (bool): Whether to use bias or not
        """
        # Add conv layers to the network as well as batchnorm and maxpool
        for i, _ in enumerate(features[:-1]):
            # Conv layers with BatchNorm and MaxPool
            self.features.append(
                BinarizedConv2d(features[i], features[i+1], kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device, latent_weights=self.latent_weights))
            self.features.append(
                BinarizedConv2d(features[i+1], features[i+1], kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation, bias=bias, device=self.device, latent_weights=self.latent_weights))
            self.features.append(
                torch.nn.AvgPool2d(kernel_size=2))
            self.features.append(torch.nn.BatchNorm2d(features[i+1],
                                                      affine=self.affine,
                                                      track_running_stats=self.running_stats,
                                                      device=self.device,
                                                      eps=self.bneps,
                                                      momentum=self.bnmomentum))
            self.features.append(torch.nn.Dropout2d(p=0.2))

    def _classifier_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            dropout (bool): Whether to use dropout
            bias (bool): Whether to use bias
        """
        # Create an input layer based on the output of the last layer of the feature extractor
        for i, _ in enumerate(layers[:-1]):
            # Linear layers with BatchNorm
            if self.dropout and i != 0:
                self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(BinarizedLinear(
                layers[i],
                layers[i+1],
                bias=bias,
                device=self.device,
                latent_weights=self.latent_weights))
            if self.batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(
                    layers[i+1],
                    affine=self.affine,
                    track_running_stats=self.running_stats,
                    device=self.device,
                    eps=self.bneps,
                    momentum=self.bnmomentum))

    def forward(self, x):
        """ Forward pass of DNN

        Args: 
            x (torch.Tensor): Input tensor

        Returns: 
            torch.Tensor: Output tensor

        """
        # Call forward of parent class
        return super().forward(x)

    def __repr__(self):
        return f"BNN(features={self.features},classifier={self.layers}, dropout={self.dropout}, latent_weights={self.latent_weights}, batchnorm={self.batchnorm}, bnmomentum={self.bnmomentum}, bneps={self.bneps}, device={self.device}"

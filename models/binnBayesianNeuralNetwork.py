
import torch
from .layers import *
from .deepNeuralNetwork import *


class BiNNBayesianNN(DNN):
    """ Binary Bayesian Neural Network

    Args:
        layers (list): List of layer sizes
        lambda_init (float): Initial standard deviation of the gaussian distribution of the lambda parameter
        n_samples (int): Number of samples to use for the forward pass
    """

    def __init__(self,
                 layers: list,
                 lambda_init: float = 0.1,
                 n_samples: int = 1,
                 *args,
                 **kwargs):
        self.lambda_init = lambda_init
        self.n_samples = n_samples
        super().__init__(layers, *args, **kwargs)

    def _layer_init(self, layers, bias=False):
        for i, _ in enumerate(layers[:-1]):
            # Dropout only on hidden layers
            if self.dropout and i != 0:
                layers.append(torch.nn.Dropout(p=0.2))

            # Bayesian Binary Linear Layer
            self.layers.append(BayesianBiNNLinear(
                layers[i],
                layers[i+1],
                lambda_init=self.lambda_init,
                bias=bias,
                device=self.device,
            ))

            # Batchnorm
            if self.batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(
                    layers[i+1],
                    affine=True,
                    track_running_stats=self.running_stats,
                    device=self.device,
                    eps=self.bneps,
                    momentum=self.bnmomentum))

    def _weight_init(self, init='normal', std=0.01):
        pass

    def forward(self, x):
        ### FORWARD PASS ###
        unique_layers = set(type(layer) for layer in self.layers)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BayesianBiNNLinear):
                x = layer(x, self.n_samples)
            else:
                x = layer(torch.mean(x, dim=0))
            if layer is not self.layers[-1] and (i+1) % len(unique_layers) == 0:
                x = self.activation_function(x)
        # Average over samples if the last layer is a MetaBayesLinearParallel layer
        if isinstance(layer, BayesianBiNNLinear):
            x = torch.nn.functional.log_softmax(x, dim=2)
            x = torch.mean(x, dim=0)
        if self.output_function == "softmax":
            x = torch.nn.functional.softmax(x, dim=1)
        if self.output_function == "log_softmax":
            x = torch.nn.functional.log_softmax(x, dim=1)
        if self.output_function == "sigmoid":
            x = torch.nn.functional.sigmoid(x)
        return x


import torch
from .layers import *
from .deepNeuralNetwork import *


class BayesianNN(DNN):
    """ Bayesian Neural Network (BNN)

    Neural Network with probabilistic weights

    Blundell et al., 2015: Weight Uncertainty in Neural Networks
    """

    def __init__(self, layers, sigma_init=0.1, n_samples=1, *args, **kwargs):
        self.sigma_init = sigma_init
        self.n_samples = n_samples
        super().__init__(layers, *args, **kwargs)

    def _layer_init(self, layers, bias=False):
        for i in range(self.n_layers+1):
            # Linear layers with BatchNorm
            if self.dropout and i != 0:
                layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(MetaBayesLinearParallel(
                layers[i], layers[i+1], bias=bias, device=self.device, sigma_init=self.sigma_init))
            if self.batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(
                    layers[i+1], affine=not bias, track_running_stats=True, device=self.device))

    def _weight_init(self, init='normal', std=0.01):
        pass

    def forward(self, x, log=True):
        ### FORWARD PASS ###
        unique_layers = set(type(layer) for layer in self.layers)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MetaBayesLinearParallel):
                x = layer(x, self.n_samples)
            # if its batchnorm, we need to average over n_samples
            else:
                x = layer(torch.mean(x, dim=0))
            if layer is not self.layers[-1] and (i+1) % len(unique_layers) == 0:
                x = torch.nn.functional.relu(x)
        # Average over samples if the last layer is a MetaBayesLinearParallel layer
        if isinstance(layer, MetaBayesLinearParallel):
            x = torch.nn.functional.log_softmax(x, dim=2)
            if not log:
                x = torch.exp(x)
            return x.mean(0)
        else:
            return torch.nn.functional.log_softmax(x, dim=1)


import torch
from .layers import *
from .deepNeuralNetwork import *


class BayesianNN(DNN):
    """ Bayesian Neural Network (BNN)

    Neural Network with probabilistic weights

    Blundell et al., 2015: Weight Uncertainty in Neural Networks
    """

    def _layer_init(self, layers, dropout=False, bias=False):
        for i in range(self.n_layers+1):
            # Linear layers with BatchNorm
            if dropout and i != 0:
                layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(MetaBayesLinearParallel(
                layers[i], layers[i+1], bias=bias, device=self.device))
            self.layers.append(torch.nn.BatchNorm1d(
                layers[i+1], affine=not bias, track_running_stats=True, device=self.device))

    def _weight_init(self, init='normal', std=0.01):
        pass

    def forward(self, x, samples=1):
        """Forward propagation of the binarized neural network
        Uses Sign activation function for binarization
        """
        for layer in self.layers:
            if isinstance(layer, MetaBayesLinearParallel):
                x = layer(x, samples)
            else:
                # transform the output of the previous layer [samples, n_neurons, n_features] to [n_neurons, n_features] by averaging over the samples
                x = torch.mean(x, dim=0)
            if layer is not self.layers[-1] and isinstance(layer, torch.nn.BatchNorm1d):
                x = torch.nn.functional.relu(x)
        return torch.nn.functional.log_softmax(x, dim=1)

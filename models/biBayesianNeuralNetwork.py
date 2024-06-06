
import torch
from .layers import *
from .deepNeuralNetwork import *


class BiBayesianNN(DNN):
    """ Neural Network Base Class
    """

    def __init__(self,
                 layers,
                 n_samples_forward: int = 1,
                 n_samples_backward: int = 1,
                 tau: float = 1,
                 binarized: bool = False,
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            n_samples_forward (int): Number of forward samples
            n_samples_backward (int): Number of backward samples
        """
        self.tau = tau
        self.n_samples_forward = n_samples_forward
        self.n_samples_backward = n_samples_backward
        self.binarized = binarized
        super().__init__(layers, *args, **kwargs)

    def _layer_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            dropout (bool): Whether to use dropout
            bias (bool): Whether to use bias
        """
        for i, _ in enumerate(layers[:-1]):
            # BiBayesian layers with BatchNorm
            if self.dropout and i != 0:
                self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(BiBayesianLinear(
                layers[i],
                layers[i+1],
                tau=self.tau,
                binarized=self.binarized,
                device=self.device))
            self._batch_norm_init(layers, i)

    def _weight_init(self, init='normal', std=0.1):
        """ Initialize weights of each layer

        Args:
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
        """
        for layer in self.layers:
            if isinstance(layer, torch.nn.Module) and hasattr(layer, 'lambda_') and layer.lambda_ is not None:
                if init == 'gaussian':
                    torch.nn.init.normal_(
                        layer.lambda_.data, mean=0.0, std=std)
                elif init == 'uniform':
                    torch.nn.init.uniform_(
                        layer.lambda_.data, a=-std/2, b=std/2)
                elif init == 'xavier':
                    torch.nn.init.xavier_normal_(layer.lambda_.data)

    def forward(self, x, backwards=True):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        # Flatten input if necessary
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        unique_layers = set(type(layer) for layer in self.layers)
        ### FORWARD PASS ###
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BiBayesianLinear):
                if backwards:
                    x = layer(x, self.n_samples_backward)
                else:
                    x = layer.sample(x, self.n_samples_forward)
            else:
                # Normalization layers, but input is (n_samples, batch, features)
                shape = x.shape
                x = x.reshape([shape[0]*shape[1], shape[2]])
                x = layer(x)
                x = x.reshape([shape[0], shape[1], shape[2]])
            if layer is not self.layers[-1] and (i+1) % len(unique_layers) == 0:
                x = self.activation_function(x)
        if self.output_function == "softmax":
            x = torch.nn.functional.softmax(x, dim=2)
        elif self.output_function == "log_softmax":
            x = torch.nn.functional.log_softmax(x, dim=2)
        elif self.output_function == "sigmoid":
            x = torch.nn.functional.sigmoid(x, dim=1)
        return x.mean(0)

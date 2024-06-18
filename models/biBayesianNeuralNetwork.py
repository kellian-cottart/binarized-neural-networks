
import torch
from .layers import *
from .deepNeuralNetwork import *
from .layers.activation import *


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
        self.layers.append(nn.Flatten(start_dim=2).to(self.device))
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
            if self.squared_inputs == True:
                self.layers.append(SquaredActivation().to(self.device))
            self.layers.append(self._norm_init(layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(self._activation_init())
                # self.layers.append(self._norm_init(layers[i+1]))

    def forward(self, x, backwards=True):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
        ### FORWARD PASS ###
        for layer in self.layers:
            if isinstance(layer, BiBayesianLinear):
                if backwards:
                    x = layer(x, self.n_samples_backward)
                else:
                    x = layer.sample(x, self.n_samples_forward)
            else:
                try:
                    x = layer(x)
                except:
                    # Normalization layers, but input is (n_samples, batch, features)
                    shape = x.shape
                    x = x.reshape([shape[0]*shape[1], shape[2]])
                    x = layer(x)
                    x = x.reshape([shape[0], shape[1], shape[2]])
        if self.output_function == "softmax":
            x = torch.nn.functional.softmax(x, dim=2)
        elif self.output_function == "log_softmax":
            x = torch.nn.functional.log_softmax(x, dim=2)
        elif self.output_function == "sigmoid":
            x = torch.nn.functional.sigmoid(x, dim=1)
        return x

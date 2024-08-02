
import torch
from .layers import *
from .deepNeuralNetwork import *
from .layers.activation import *


class BayesianNN(DNN):
    """ Neural Network Base Class
    """

    def __init__(self,
                 layers,
                 zeroMean=False,
                 std=0.1,
                 n_samples_forward=1,
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            n_samples_forward (int): Number of forward samples
            n_samples_backward (int): Number of backward samples
        """
        self.zeroMean = zeroMean
        self.sigma_init = std
        self.n_samples_forward = n_samples_forward
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
            self.layers.append(MetaBayesLinearParallel(
                in_features=layers[i],
                out_features=layers[i+1],
                bias=bias,
                zeroMean=self.zeroMean,
                sigma_init=self.sigma_init,
                device=self.device
            ))
            self.layers.append(self._norm_init(layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(self._activation_init())

    def _weight_init(self, init='normal', std=0.1):
        pass

    def forward(self, x, *args, **kwargs):
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
            if isinstance(layer, MetaBayesLinearParallel):
                x = layer(x, self.n_samples_forward)
            else:
                try:
                    x = layer(x)
                except:
                    # Normalization layers, but input is (n_samples, batch, features)
                    shape = x.shape
                    x = x.reshape([shape[0]*shape[1], shape[2]])
                    x = layer(x)
                    x = x.reshape([shape[0], shape[1], shape[2]])
        return x

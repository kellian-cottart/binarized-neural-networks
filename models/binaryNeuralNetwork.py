
from torch.nn import Sequential, Flatten, Dropout
from .layers import *
from .deepNeuralNetwork import *


class BiNN(DNN):
    """ Binarized Neural Network (BiNN)

    Neural Network with binary weights and activations, using hidden weights called "degrees of certainty" (DOCs) to approximate real-valued weights.

    Args:
        layers (list): List of layer sizes (including input and output layers)
        init (str): Initialization method for weights
        std (float): Standard deviation for initialization
        device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
        dropout (bool): Whether to use dropout
    """

    def _layer_init(self, layers, bias=False):
        self.layers = Sequential(*self.layers)
        self.layers.append(Flatten().to(self.device))
        for i, _ in enumerate(layers[:-1]):
            # Linear layers with BatchNorm
            if self.dropout and i != 0:
                self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(BinarizedLinear(
                layers[i],
                layers[i+1],
                bias=bias,
                device=self.device))
            self.layers.append(self._norm_init(layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(self._activation_init())
            if self.dropout and i < len(layers)-2:
                self.layers.append(Dropout(p=0.5))

    def __repr__(self):
        return f"BNN({self.layers}, dropout={self.dropout}, device={self.device}, normalization={self.normalization})"

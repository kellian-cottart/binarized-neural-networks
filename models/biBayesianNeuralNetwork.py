
import torch
from .layers import *
from .deepNeuralNetwork import *
from .layers.activation import *


class BiBayesianNN(DNN):
    """ Neural Network Base Class
    """

    def __init__(self,
                 n_samples_test: int = 1,
                 n_samples_train: int = 1,
                 tau: float = 1,
                 binarized: bool = False,
                 classifier: bool = False,
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            n_samples_test (int): Number of forward samples
            n_samples_train (int): Number of backward samples
        """
        self.tau = tau
        self.n_samples_test = n_samples_test
        self.n_samples_train = n_samples_train
        self.binarized = binarized
        self.classifier = classifier
        super().__init__(*args, **kwargs)
        self.layers = BinarizedSequential(*self.layers)

    def _layer_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            dropout (bool): Whether to use dropout
            bias (bool): Whether to use bias
        """
        self.layers.append(nn.Flatten(start_dim=1).to(self.device))
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

    def forward(self, x, backwards=True):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        samples = self.n_samples_train if backwards else self.n_samples_test
        if x.dim() == 4 and hasattr(self, "classifier"):
            x = x.repeat(samples, *(1,)*len(x.size()[1:]))
        ### FORWARD PASS ###
        x = self.layers(x, backwards=backwards,
                        samples=samples)
        return x.reshape(samples, x.size(0)//samples, *(x.size()[1:]))


class BinarizedSequential(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, samples, backwards=True):
        for layer in self:
            if isinstance(layer, BiBayesianLinear):
                if backwards:
                    x = layer(x, samples)
                else:
                    x = layer.sample(x, samples)
            elif isinstance(layer, InstanceNorm1d):
                x = x.unsqueeze(1)
                x = layer(x)
                x = x.squeeze(1)
            else:
                x = layer(x)
        return x

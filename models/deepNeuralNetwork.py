import copy
import torch
from .layers import *


class StandardizeNorm(torch.nn.Module):
    """ 0 mean, 1 variance normalization
    This is exactly the same as instance normalization
    """

    def __init__(self, eps=1e-5):
        super(StandardizeNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + self.eps)


class DNN(torch.nn.Module):
    """ Neural Network Base Class
    """

    def __init__(self,
                 layers: list = [1024, 1024],
                 init: str = "uniform",
                 std: float = 0.01,
                 device: str = "cuda:0",
                 dropout: bool = False,
                 normalization: str = None,
                 bias: bool = False,
                 running_stats: bool = False,
                 affine: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.15,
                 gnnum_groups: int = 32,
                 activation_function: torch.nn.functional = torch.nn.functional.relu,
                 output_function: str = "softmax",
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            layers (list): List of layer sizes (including input and output layers)
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
            device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
            dropout (bool): Whether to use dropout
            normalization (str): Normalization method to choose (e.g. 'batchnorm', 'layernorm', 'instancenorm', 'groupnorm')
            bias (bool): Whether to use bias
            eps (float): BatchNorm epsilon
         (float): BatchNorm momentum
            running_stats (bool): Whether to use running stats in BatchNorm
            affine (bool): Whether to use affine transformation in BatchNorm
            gnnum_groups (int): Number of groups in GroupNorm
            activation_function (torch.nn.functional): Activation function
            output_function (str): Output function
        """
        super(DNN, self).__init__()
        self.device = device
        self.layers = torch.nn.ModuleList().to(self.device)
        self.dropout = dropout
        self.normalization = normalization
        self.eps = eps
        self.momentum = momentum
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.output_function = output_function
        self.gnnum_groups = gnnum_groups
        ### LAYER INITIALIZATION ###
        self._layer_init(layers, bias)
        ### WEIGHT INITIALIZATION ###
        self._weight_init(init, std)

    def _layer_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            dropout (bool): Whether to use dropout
            bias (bool): Whether to use bias
        """
        self.layers.append(nn.Flatten().to(self.device))
        for i, _ in enumerate(layers[:-1]):
            # Linear layers with BatchNorm
            if self.dropout and i != 0:
                self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.Linear(
                layers[i],
                layers[i+1],
                bias=bias,
                device=self.device))
            self.layers.append(self._norm_init(layers[i+1]))

    def _norm_init(self, n_features):
        """ Initialize normalization layers"""
        if self.normalization == "batchnorm":
            return torch.nn.BatchNorm1d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats).to(self.device)
        elif self.normalization == "layernorm":
            return torch.nn.LayerNorm(n_features).to(self.device)
        elif self.normalization == "instancenorm":
            return torch.nn.InstanceNorm1d(n_features, eps=self.eps, affine=self.affine, track_running_stats=self.running_stats).to(self.device)
        elif self.normalization == "groupnorm":
            return torch.nn.GroupNorm(self.gnnum_groups, n_features).to(self.device)
        elif self.normalization == "standardize":
            return StandardizeNorm().to(self.device)
        else:
            return torch.nn.Identity().to(self.device)

    def _weight_init(self, init='normal', std=0.1):
        """ Initialize weights of each layer

        Args:
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
        """
        for layer in self.layers:
            if isinstance(layer, torch.nn.Module) and hasattr(layer, 'weight') and layer.weight is not None:
                if init == 'gaussian':
                    torch.nn.init.normal_(
                        layer.weight.data, mean=0.0, std=std)
                elif init == 'uniform':
                    torch.nn.init.uniform_(
                        layer.weight.data, a=-std/2, b=std/2)
                elif init == 'xavier':
                    torch.nn.init.xavier_normal_(layer.weight.data)

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        unique_layers = set(type(layer) for layer in self.layers)
        ### FORWARD PASS ###
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

    def load_bn_states(self, state_dict):
        """ Load batch normalization states

        Args:
            state_dict (dict): State dictionary

        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.load_state_dict(state_dict[f"layers.{i}"])

    def save_bn_states(self):
        """ Save batch normalization states

        """
        state_dict = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.BatchNorm1d):
                state_dict[f"layers.{i}"] = copy.deepcopy(layer.state_dict())
        return state_dict

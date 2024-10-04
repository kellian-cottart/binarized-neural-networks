import copy
from .layers import *
from .layers.activation import *
from torch.nn import InstanceNorm1d


class DNN(torch.nn.Module):
    """ Deep Neural Network

    Args:
        layers (list): List of layer sizes (including input and output layers)
        init (str): Initialization method for weights
        std (float): Standard deviation for initialization
        device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
        dropout (bool): Whether to use dropout
        normalization (str): Normalization method to choose (e.g. 'batchnorm', 'layernorm', 'instancenorm', 'groupnorm')
        bias (bool): Whether to use bias
        eps (float): BatchNorm epsilon
        momentum (float): BatchNorm momentum
        running_stats (bool): Whether to use running stats in BatchNorm
        affine (bool): Whether to use affine transformation in BatchNorm
        gnnum_groups (int): Number of groups in GroupNorm
        activation_function (torch.nn.functional): Activation function
        output_function (str): Output function
        squared_inputs (bool): Whether to square the inputs
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
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
                 activation_function: str = "relu",
                 squared_inputs: bool = False,
                 *args,
                 **kwargs):
        super(DNN, self).__init__()
        self.device = device
        self.dropout = dropout
        self.normalization = normalization
        self.eps = eps
        self.momentum = momentum
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.gnnum_groups = gnnum_groups
        self.squared_inputs = squared_inputs
        self.std = std
        self.layers = []
        if "activation_parameters" in kwargs:
            self.activation_parameters = kwargs["activation_parameters"]
        ### LAYER INITIALIZATION ###
        self._layer_init(layers, bias)

    def _layer_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            layers (list): List of layer sizes (including input and output layers)
            bias (bool): Whether to use bias
        """
        self.layers.append(Flatten().to(self.device))
        for i, _ in enumerate(layers[:-1]):
            self.layers.append(torch.nn.Linear(
                layers[i],
                layers[i+1],
                bias=bias,
                device=self.device))
            if self.squared_inputs == True:
                self.layers.append(SquaredActivation().to(self.device))
            self.layers.append(self._norm_init(layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(self._activation_init())
            if self.dropout and i < len(layers)-2:
                self.layers.append(torch.nn.Dropout(p=0.5))
        self.layers = torch.nn.Sequential(*self.layers).to(self.device)

    def load_bn_states(self, state_dict):
        """ Load batch normalization states

        Args:
            state_dict (dict): State dictionary

        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.BatchNorm1d or torch.nn.InstanceNorm1d):
                layer.load_state_dict(state_dict[f"layers.{i}"])

    def save_bn_states(self):
        """ Save batch normalization states

        Returns:
            dict: State dictionary
        """
        state_dict = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.BatchNorm1d or torch.nn.InstanceNorm1d):
                state_dict[f"layers.{i}"] = copy.deepcopy(layer.state_dict())
        return state_dict

    def _activation_init(self):
        """
        Returns:
            torch.nn.Module: Activation function module
        """
        activation_functions = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "silu": torch.nn.SiLU,
            "softplus": torch.nn.Softplus,
            "tanh": torch.nn.Tanh,
            "sign": SignActivation,
            "squared": SquaredActivation,
            "elephant": ElephantActivation,
            "gate": GateActivation,
            "gaussian": GaussianActivation,
            "hyperbolic": HyperbolicCosineActivation,
        }
        # add parameters to activation function if needed
        try:
            return activation_functions.get(self.activation_function, torch.nn.Identity)(**self.activation_parameters).to(self.device)
        except:
            return activation_functions.get(self.activation_function, torch.nn.Identity)().to(self.device)

    def _norm_init(self, n_features):
        """
        Args:
            n_features (int): Number of features

        Returns:
            torch.nn.Module: Normalization layer module
        """
        normalization_layers = {
            "batchnorm": lambda: torch.nn.BatchNorm1d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats),
            "layernorm": lambda: torch.nn.LayerNorm(n_features),
            "instancenorm": lambda: torch.nn.InstanceNorm1d(1, eps=self.eps, affine=self.affine, track_running_stats=self.running_stats),
            "groupnorm": lambda: torch.nn.GroupNorm(self.gnnum_groups, n_features),
        }
        return normalization_layers.get(self.normalization, torch.nn.Identity)().to(self.device)

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        ### FORWARD PASS ###
        for layer in self.layers:
            if isinstance(layer, InstanceNorm1d):
                x = x.unsqueeze(1)
                x = layer(x)
                x = x.squeeze(1)
            else:
                x = layer(x)
        return x

    def _extra_repr(self):
        return f"layers={self.layers}, activation_function={self.activation_function}, normalization={self.normalization}, dropout={self.dropout}, device={self.device}"

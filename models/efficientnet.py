import torch
from torch.nn import *
from .deepNeuralNetwork import DNN
from .layers.activation import *
import torchvision

LOOK_UP_DICT = {}
for i in range(8):
    LOOK_UP_DICT[str(i)] = {
        "model": getattr(torchvision.models, f"efficientnet_b{i}"),
        "weights": getattr(torchvision.models, f"EfficientNet_B{i}_Weights")
    }


class EfficientNet(Module):
    """ EfficientNet Neural Network
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
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
                 activation_function: str = "relu",
                 gnnum_groups: int = 32,
                 frozen=False,
                 version=0,
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            layers (list): List of layer sizes for the classifier
            features (list): List of layer sizes for the feature extractor
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
            device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
            dropout (bool): Whether to use dropout
            batchnorm (bool): Whether to use batchnorm
            bias (bool): Whether to use bias
            bneps (float): BatchNorm epsilon
            bnmomentum (float): BatchNorm momentum
            running_stats (bool): Whether to use running stats in BatchNorm
            affine (bool): Whether to use affine transformation in BatchNorm
            activation_function (torch.nn.functional): Activation function
            output_function (str): Output function
        """
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.normalization = normalization
        self.eps = eps
        self.momentum = momentum
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.gnnum_groups = gnnum_groups
        self.version = version
        # retrieve weights from EfficientNet
        current = LOOK_UP_DICT[str(version)]
        effnet = current["model"](weights=current["weights"].DEFAULT)
        # remove classifier layers
        self.features = torch.nn.ModuleList(
            list(effnet.features.children())).to(self.device)
        # freeze feature extractor
        if frozen == True:
            for param in self.features.parameters():
                param.requires_grad = False
                param.grad = None
        ## CLASSIFIER INITIALIZATION ##
        self.classifier = DNN(layers, init, std, device, dropout, normalization, bias,
                              running_stats, affine, eps, momentum, gnnum_groups, activation_function)

    def _activation_init(self):
        """
        Returns:
            torch.nn.Module: Activation function module
        """
        activation_functions = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "tanh": torch.nn.Tanh,
            "sign": SignActivation,
            "squared": SquaredActivation,
            "elephant": ElephantActivation,
            "gate": GateActivation
        }
        # add parameters to activation function if needed
        try:
            return activation_functions.get(self.activation_function, torch.nn.Identity)(**self.activation_parameters).to(self.device)
        except:
            return activation_functions.get(self.activation_function, torch.nn.Identity)().to(self.device)

    def _norm_init(self, n_features):
        """Returns a layer of normalization"""
        if self.normalization == "batchnorm":
            return torch.nn.BatchNorm2d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats).to(self.device)
        elif self.normalization == "layernorm":
            return torch.nn.LayerNorm(n_features).to(self.device)
        elif self.normalization == "instancenorm":
            return torch.nn.InstanceNorm2d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats).to(self.device)
        elif self.normalization == "groupnorm":
            return torch.nn.GroupNorm(self.gnnum_groups, n_features).to(self.device)
        else:
            return torch.nn.Identity().to(self.device)

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        for layer in self.features:
            x = layer(x)
        return self.classifier.forward(x)

    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return super().extra_repr() + f"version={self.version}, params={self.number_parameters()}"

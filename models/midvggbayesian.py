import torch
from torch.nn import *
from .bayesianNeuralNetwork import BayesianNN
from .layers import *
from .layers.activation import *


class MidVGGBayesian(Module):
    """ Convolutional Neural Network Base Class
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
                 n_samples_forward: int = 1,
                 zeroMean=False,
                 bayesian_convolution=True,
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
        self.bias = bias
        self.sigma_init = std
        self.activation_function = activation_function
        self.n_samples_forward = n_samples_forward
        # retrieve weights from VGG16
        vgg16 = torch.hub.load('pytorch/vision:v0.9.0',
                               'vgg16', pretrained=True)
        # freeze weights
        if bayesian_convolution == True:
            self.features = self.make_vgg16()
        else:
            self.features = torch.nn.ModuleList(
                list(vgg16.features.children())).to(self.device)
            for param in self.features.parameters():
                param.requires_grad = False
                param.grad = None
        ## CLASSIFIER INITIALIZATION ##
        self.classifier = BayesianNN(layers=layers,
                                     n_samples_forward=n_samples_forward,
                                     zeroMean=zeroMean,
                                     sigma_init=std,
                                     device=device,
                                     dropout=dropout,
                                     init=init,
                                     std=std,
                                     normalization=normalization,
                                     bias=bias,
                                     running_stats=running_stats,
                                     affine=affine,
                                     eps=eps,
                                     momentum=momentum,
                                     gnnum_groups=gnnum_groups,
                                     activation_function=activation_function
                                     )

    def make_vgg16(self):
        # save vgg16 features
        vgg16 = torch.hub.load('pytorch/vision:v0.9.0',
                               'vgg16', pretrained=True)
        features = torch.nn.ModuleList(vgg16.features.children())
        # iterate on every feature and replace with bayesian layer
        for layer in features:
            if isinstance(layer, torch.nn.Conv2d):
                weights = layer.weight.data
                bias = layer.bias.data if layer.bias is not None else None
                layer = MetaBayesConv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size[0], stride=layer.stride[0],
                                        padding=layer.padding[0], bias=self.bias, sigma_init=self.sigma_init*1e-3, device=self.device)
                layer.weight_mu.data = weights
                if bias is not None:
                    layer.bias_mu.data = bias
        return features

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
            if isinstance(layer, MetaBayesConv2d):
                x = layer(
                    x, self.n_samples_forward if self.n_samples_forward > 1 else 1)
            else:
                try:
                    x = layer(x)
                except:
                    # Normalization layers, but input is (n_samples, batch, features)
                    shape = x.shape
                    x = x.reshape([shape[0]*shape[1], *x.shape[2:]])
                    x = layer(x)
                    x = x.reshape([shape[0], shape[1], *x.shape[1:]])
        return self.classifier.forward(x)

    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return super().extra_repr() + f"parameters={self.number_parameters()}"

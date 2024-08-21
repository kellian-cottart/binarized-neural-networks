import torch
from torch.nn import *
from .bayesianNeuralNetwork import BayesianNN
from .layers import *
from .layers.activation import *
from torchvision.models import vgg16, VGG16_Weights
import torchvision


LOOK_UP_DICT = {}
for i in range(8):
    LOOK_UP_DICT[str(i)] = {
        "model": getattr(torchvision.models, f"efficientnet_b{i}"),
        "weights": getattr(torchvision.models, f"EfficientNet_B{i}_Weights")
    }


class EfficientNetBayesian(Module):
    """ EfficientNet Bayesian Neural Network
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
                 n_samples_train: int = 1,
                 zeroMean=False,
                 frozen=False,
                 sigma_multiplier=1,
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
        self.bias = bias
        self.sigma_init = std
        self.activation_function = activation_function
        self.n_samples_train = n_samples_train
        self.sigma_multiplier = sigma_multiplier
        self.version = version
        # retrieve weights from EfficientNet
        current = LOOK_UP_DICT[str(version)]
        effnet = current["model"](weights=current["weights"].IMAGENET1K_V1)
        self.features = torch.nn.ModuleList(
            list(effnet.features.children())).to(self.device)
        self.features = self.replace_conv(self.features)
        self.avgpool = effnet.avgpool
        if frozen == True:
            for param in self.features.parameters():
                param.requires_grad = False
                param.grad = None

        ## CLASSIFIER INITIALIZATION ##
        self.classifier = BayesianNN(
            layers=layers,
            n_samples_train=n_samples_train,
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

    def conv_to_bayesian(self, layer):
        new_layer = MetaBayesConv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size[0], stride=layer.stride[0],
                                    padding=layer.padding[0], bias=layer.bias is not None, sigma_init=self.sigma_init*self.sigma_multiplier, device=self.device, groups=layer.groups)
        new_layer.weight_mu.data = layer.weight.data.clone()
        if layer.bias is not None:
            new_layer.bias_mu.data = layer.bias.data.clone()
        return new_layer

    def replace_excitation(self, layer):
        """ Replace torchvision's SqueezeExcitation layer with a Bayesian one, by turning the whole layer into a ModuleList
        https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py
        """
        new_layer = torch.nn.ModuleList()
        new_layer.append(torch.nn.AdaptiveAvgPool2d(1))
        new_layer.append(self.conv_to_bayesian(layer.fc1))
        new_layer.append(self.conv_to_bayesian(layer.fc2))
        new_layer.append(torch.nn.SiLU(inplace=True))
        new_layer.append(torch.nn.Sigmoid())
        return new_layer

    def replace_convNorm(self, layer):
        """ Replace torchvision's ConvNorm layer with a Bayesian one, by turning the whole layer into a ModuleList
        """
        new_module = torch.nn.ModuleList(layer.children())
        for k, elem in enumerate(new_module):
            if isinstance(elem, torch.nn.Conv2d):
                new_module[k] = self.conv_to_bayesian(elem)
        return new_module

    def replace_mbconv(self, layer):
        """ Replace torchvision's MBConv layer with a Bayesian one, by turning the whole layer into a ModuleList
        https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
        """
        # layer.block is a sequential that we turn into a ModuleList
        new_layer = torch.nn.ModuleList(layer.block.children())
        # replace the Conv2d layers
        for i, iterable in enumerate(new_layer):
            if "NormActivation" in iterable.__class__.__name__:
                new_layer[i] = self.replace_convNorm(iterable)
            elif "SqueezeExcitation" in iterable.__class__.__name__:
                new_layer[i] = self.replace_excitation(iterable)
        return new_layer

    def replace_conv(self, module_list):
        # our goal is to replace every Conv2d layer with a BayesianConv2d layer
        # module_list is a torch.nn.ModuleList that can contain other ModuleLists or Sequential objects
        # we need to iterate over all of them and replace the Conv2d layers
        if not isinstance(module_list, torch.nn.ModuleList):
            module_list = torch.nn.ModuleList(module_list.children())
        for i, layer in enumerate(module_list):
            if isinstance(layer, torch.nn.Conv2d):
                # replace the layer
                module_list[i] = self.conv_to_bayesian(layer)
            # elif iterable
            elif "MBConv" in layer.__class__.__name__:
                module_list[i] = self.replace_mbconv(layer)
            elif "NormActivation" in layer.__class__.__name__:
                module_list[i] = self.replace_convNorm(layer)
            else:
                try:
                    iterator = iter(layer)
                except TypeError:
                    continue
                else:
                    module_list[i] = self.replace_conv(layer)

        # Now, we will unpack all module lists into a single module list
        while any(isinstance(layer, torch.nn.ModuleList) for layer in module_list):
            new_module_list = torch.nn.ModuleList()
            for layer in module_list:
                if isinstance(layer, torch.nn.ModuleList):
                    new_module_list.extend(layer)
                else:
                    new_module_list.append(layer)
            module_list = new_module_list

        return module_list

    def layer_forward(self, layer, x, *args, **kwargs):
        if isinstance(layer, MetaBayesConv2d):
            x = layer(
                x, self.n_samples_train if self.n_samples_train > 1 else 1)
        else:
            try:
                x = layer(x)
            except:
                # Normalization layers, but input is (n_samples, batch, features)
                shape = x.shape
                x = x.reshape([shape[0]*shape[1], *x.shape[2:]])
                x = layer(x)
                x = x.reshape([shape[0], shape[1], *x.shape[1:]])
        return x

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        for layer in self.features:
            x = self.layer_forward(layer, x)
        x = self.avgpool(x)
        return self.classifier.forward(x)

    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return super().extra_repr() + f"n_samples_train={self.n_samples_train}, sigma_init={self.sigma_init}, sigma_multiplier={self.sigma_multiplier}, version={self.version}, params={self.number_parameters()}"

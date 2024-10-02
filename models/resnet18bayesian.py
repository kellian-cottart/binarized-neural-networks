from torch.nn import *
from .bayesianNeuralNetwork import BayesianNN
from .layers.activation import *
import torchvision
from .layers import *
from torchvision.models.resnet import BasicBlock


class ResNet18Hybrid(Module):
    """ ResNet18 Neural Network
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
                 n_samples_train: int = 1,
                 zeroMean=False,
                 sigma_multiplier=1,
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
        self.n_samples_train = n_samples_train
        self.sigma_multiplier = sigma_multiplier
        self.sigma_init = std
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # remove classifier layers
        self.features = MetaBayesSequential(*list(resnet.children())[:-1])

        self.features = self._replace_layers(self.features)
        # freeze feature extractor
        if frozen == True:
            for param in self.features.parameters():
                param.requires_grad = False
                param.grad = None

        self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        layers.insert(0, list(resnet.children())[-1].in_features)
        ## CLASSIFIER INITIALIZATION ##
        self.classifier = BayesianNN(layers=layers,
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
                                     activation_function=activation_function,
                                     classifier=False,
                                     )

    def _replace_conv2d(self, layer):
        """Replace Conv2d with MetaBayesConv2d"""
        new_layer = MetaBayesConv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size,
                                    stride=layer.stride, padding=layer.padding, bias=layer.bias is not None,
                                    sigma_init=self.sigma_init * self.sigma_multiplier, device=self.device,
                                    groups=layer.groups)
        new_layer.weight.mu.data = layer.weight.data.clone()
        if layer.bias is not None:
            new_layer.bias.mu.data = layer.bias.data.clone()
        return new_layer

    def _replace_batchnorm2d(self, layer):
        """Replace BatchNorm2d with MetaBayesBatchNorm2d"""
        new_layer = MetaBayesBatchNorm2d(num_features=layer.num_features, eps=layer.eps,
                                         sigma_init=self.sigma_init * self.sigma_multiplier,
                                         momentum=layer.momentum, affine=layer.affine,
                                         track_running_stats=layer.track_running_stats)
        if layer.affine:
            new_layer.weight.mu.data = layer.weight.data.clone()
            new_layer.bias.mu.data = layer.bias.data.clone()
        if layer.track_running_stats:
            new_layer.running_mean.data = layer.running_mean.data.clone()
            new_layer.running_var.data = layer.running_var.data.clone()
            new_layer.num_batches_tracked.data = layer.num_batches_tracked.data.clone()
        return new_layer

    def _replace_layers(self, module):
        """Recursively replace Conv2d, BatchNorm2d, Sequential, and BasicBlock with their Bayesian counterparts."""

        # Check if the module itself is a BasicBlock
        if isinstance(module, BasicBlock):
            # Replace BasicBlock with MetaBayesBasicBlock
            new_block = MetaBayesBasicBlock(
                inplanes=module.conv1.in_channels, planes=module.conv1.out_channels, norm_layer=module.bn1.__class__
            )
            # Replace layers inside the BasicBlock
            new_block.conv1 = self._replace_conv2d(module.conv1)
            new_block.bn1 = self._replace_batchnorm2d(module.bn1)
            new_block.conv2 = self._replace_conv2d(module.conv2)
            new_block.bn2 = self._replace_batchnorm2d(module.bn2)
            new_block.downsample = MetaBayesSequential(
                self._replace_conv2d(module.downsample[0]),
                self._replace_batchnorm2d(module.downsample[1]),
            ) if module.downsample is not None else None
            return new_block

        # Now iterate over named children and replace layers as before
        for name, layer in module.named_children():
            if isinstance(layer, Conv2d):
                # Replace Conv2d with MetaBayesConv2d
                new_layer = self._replace_conv2d(layer)
                setattr(module, name, new_layer)
            elif isinstance(layer, BatchNorm2d):
                # Replace BatchNorm2d with MetaBayesBatchNorm2d
                new_layer = self._replace_batchnorm2d(layer)
                setattr(module, name, new_layer)
            elif isinstance(layer, Sequential):
                # Replace Sequential with MetaBayesSequential
                new_layer = MetaBayesSequential(
                    *[self._replace_layers(sub_layer) for sub_layer in layer]
                )
                setattr(module, name, new_layer)
            else:
                # Recursively replace layers in submodules if applicable
                self._replace_layers(layer)
        return module

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        x = self.transform(x)
        x = self.features(x, 0)
        return self.classifier(x)

    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return super().extra_repr() + f"params={self.number_parameters()}"

from torch.nn import Module, BatchNorm2d, Conv2d, ModuleList
from torchvision import models
from torchvision.models.efficientnet import MBConvConfig
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from .layers import *
from .bayesianNeuralNetwork import *
from functools import partial

LOOK_UP_DICT = {}
for i in range(8):
    LOOK_UP_DICT[str(i)] = {
        "model": getattr(models, f"efficientnet_b{i}"),
        "weights": getattr(models, f"EfficientNet_B{i}_Weights")
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
        self.std = std
        # retrieve weights from EfficientNet
        current = LOOK_UP_DICT[str(version)]
        effnet = current["model"](weights=current["weights"].IMAGENET1K_V1)
        effnet.features.append(effnet.avgpool)
        self.features = self.replace_conv(effnet.features)
        self.features = MetaBayesSequential(*self.features)
        self.transform = current["weights"].IMAGENET1K_V1.transforms()
        # append features to add avgpool
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
            activation_function=activation_function,
            classifier=True,
        )

    def conv_to_bayesian(self, layer):
        new_layer = MetaBayesConv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride,
                                    padding=layer.padding, bias=layer.bias is not None, sigma_init=self.std*self.sigma_multiplier, device=self.device, groups=layer.groups)
        new_layer.weight.mu.data = layer.weight.data.clone()
        if layer.bias is not None:
            new_layer.bias.mu.data = layer.bias.data.clone()
        return new_layer

    def replace_excitation(self, layer):
        """ Replace torchvision's SqueezeExcitation layer's convolutions with Bayesian ones
        https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py
        """
        new_excitation = MetaBayesSqueezeExcitation(
            layer.fc1.in_channels,
            layer.fc1.out_channels,
            layer.activation,
            layer.scale_activation,
            sigma_init=self.std*self.sigma_multiplier)
        new_excitation.set_weights(layer.fc1, layer.fc2)
        return new_excitation

    def replace_convNorm(self, layer):
        """ Replace torchvision's ConvNorm layer with a Bayesian one, by turning the whole layer into a ModuleList
        """
        new_sequential = ModuleList()
        for k, elem in enumerate(layer):
            if isinstance(elem, Conv2d):
                new_layer = self.conv_to_bayesian(elem)
            # BatchNorm doesn't really work well with Bayesian, so we replace it with an Identity
            elif isinstance(elem, BatchNorm2d):
                new_layer = MetaBayesBatchNorm2d(num_features=elem.num_features, eps=elem.eps, sigma_init=self.std*self.sigma_multiplier,
                                                 momentum=elem.momentum, affine=elem.affine, track_running_stats=elem.track_running_stats)
                if elem.affine == True:
                    new_layer.weight.mu.data = elem.weight.data.clone()
                    new_layer.bias.mu.data = elem.bias.data.clone()
                    if elem.track_running_stats:
                        new_layer.running_mean.data = elem.running_mean.data.clone()
                        new_layer.running_var.data = elem.running_var.data.clone()
                        new_layer.num_batches_tracked.data = elem.num_batches_tracked.data.clone()
            else:
                new_layer = elem
            new_sequential.append(new_layer)
        return MetaBayesSequential(*new_sequential)

    def replace_mbconv(self, layer, cnf):
        """ Replace torchvision's MBConv layer with a Bayesian one, by turning the whole layer into a ModuleList
        https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
        """
        # replace the block with a sequential
        new_conv = cnf.block(cnf, layer.stochastic_depth.p, BatchNorm2d)
        new_conv.load_state_dict(layer.state_dict())
        new_conv.stochastic_depth.load_state_dict(
            layer.stochastic_depth.state_dict())
        new_conv.use_res_connect = layer.use_res_connect if hasattr(
            layer, "use_res_connect") else None
        for i, iterable in enumerate(layer.block):
            if isinstance(iterable, Conv2dNormActivation):
                layer.block[i] = self.replace_convNorm(iterable)
            elif isinstance(iterable, SqueezeExcitation):
                layer.block[i] = self.replace_excitation(iterable)
        new_conv.block = MetaBayesSequential(*layer.block)
        return new_conv

    def replace_conv(self, features):
        configurations = [
            MBConvConfig(1, 3, 1, 32, 16, 1, block=MetaBayesMBConv),
            MBConvConfig(6, 3, 2, 16, 24, 2, block=MetaBayesMBConv),
            MBConvConfig(6, 5, 2, 24, 40, 2, block=MetaBayesMBConv),
            MBConvConfig(6, 3, 2, 40, 80, 3, block=MetaBayesMBConv),
            MBConvConfig(6, 5, 1, 80, 112, 3, block=MetaBayesMBConv),
            MBConvConfig(6, 5, 2, 112, 192, 4, block=MetaBayesMBConv),
            MBConvConfig(6, 3, 1, 192, 320, 1, block=MetaBayesMBConv),
        ]
        block_count = 0
        for i, layer in enumerate(features):
            if isinstance(layer, Conv2dNormActivation):
                features[i] = self.replace_convNorm(layer)
            elif isinstance(layer, Sequential):
                for k, mbconv in enumerate(layer):
                    if isinstance(mbconv, MBConv):
                        current_config = configurations[block_count]
                        if k > 0:
                            current_config = MBConvConfig(
                                expand_ratio=current_config.expand_ratio,
                                kernel=current_config.kernel,
                                stride=current_config.stride,
                                input_channels=current_config.out_channels,
                                out_channels=current_config.out_channels,
                                num_layers=current_config.num_layers,
                                block=MetaBayesMBConv
                            )
                        features[i][k] = self.replace_mbconv(
                            mbconv, current_config)
                block_count += 1
                if not isinstance(layer, MetaBayesSequential):
                    features[i] = MetaBayesSequential(*layer)
        return features

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        repeat_samples = self.n_samples_train if self.n_samples_train > 1 else 1
        samples = self.n_samples_train
        x = self.transform(x)
        x = x.repeat(repeat_samples, *([1] * (len(x.size())-1)))
        x = self.features(x, samples)
        return self.classifier(x)

    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return super().extra_repr() + f"n_samples_train={self.n_samples_train}, sigma_init={self.std}, sigma_multiplier={self.sigma_multiplier}, version={self.version}, params={self.number_parameters()}"

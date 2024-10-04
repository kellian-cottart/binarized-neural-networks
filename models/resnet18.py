from torch.nn import *
from .deepNeuralNetwork import DNN
from .layers.activation import *
import torchvision


class ResNet18(Module):
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
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # remove classifier layers
        self.features = Sequential(*list(resnet.children())[:-1])
        layers.insert(0, list(resnet.children())[-1].in_features)
        # freeze feature extractor
        if frozen == True:
            for param in self.features.parameters():
                param.requires_grad = False
                param.grad = None
        ## CLASSIFIER INITIALIZATION ##
        self.classifier = DNN(layers=layers,
                              init=init,
                              std=std,
                              device=device,
                              dropout=dropout,
                              normalization=normalization,
                              bias=bias,
                              running_stats=running_stats,
                              affine=affine,
                              eps=eps,
                              momentum=momentum,
                              activation_function=activation_function,
                              gnnum_groups=gnnum_groups,
                              classifier=True,
                              *args,
                              **kwargs)

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        x = self.features(x)
        return self.classifier(x)

    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return super().extra_repr() + f"params={self.number_parameters()}"

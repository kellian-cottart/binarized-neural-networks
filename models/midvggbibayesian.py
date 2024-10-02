from torch.nn import *
from .biBayesianNeuralNetwork import BiBayesianNN
from .layers.activation import *
from torchvision.models import vgg16, VGG16_Weights


class MidVGGBiBayesian(Module):
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
                 n_samples_train: int = 1,
                 n_samples_test: int = 1,
                 tau: float = 1.0,
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
        # retrieve weights from VGG16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # remove classifier layers
        self.features = ModuleList(
            list(vgg.features.children())).to(self.device)
        # freeze weights
        if frozen == True:
            for param in self.features.parameters():
                param.requires_grad = False
                param.grad = None
        layers.insert(0, list(vgg.children())[-1].in_features)
        ## CLASSIFIER INITIALIZATION ##
        self.classifier = BiBayesianNN(layers=layers,
                                       n_samples_train=n_samples_train,
                                       n_samples_test=n_samples_test,
                                       tau=tau,
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

    def extra_repr(self) -> str:
        return super().extra_repr() + f"parameters={self.number_parameters()}"

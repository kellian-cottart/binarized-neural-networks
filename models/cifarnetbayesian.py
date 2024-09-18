from torch.nn import Module, BatchNorm2d, Conv2d, ModuleList, ReLU, MaxPool2d
from torchvision.models.efficientnet import MBConvConfig
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from .layers import *
from .bayesianNeuralNetwork import *
from torchvision.transforms import v2


class CifarNetBayesian(Module):
    """ https://www.kaggle.com/code/farzadnekouei/cifar-10-image-classification-with-cnn#Step-4-%7C-Define-CNN-Model-Architecture
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
        # append features to add avgpool
        if frozen == True:
            for param in self.features.parameters():
                param.requires_grad = False
                param.grad = None

        self.features = MetaBayesSequential(
            MetaBayesConv2d(in_channels=3, out_channels=32, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(32, device=device, sigma_init=std, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MetaBayesConv2d(in_channels=32, out_channels=32, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(32, device=device, sigma_init=std*sigma_multiplier, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.2),
            MetaBayesConv2d(in_channels=32, out_channels=64, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(64, device=device, sigma_init=std*sigma_multiplier, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MetaBayesConv2d(in_channels=64, out_channels=64, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(64, device=device, sigma_init=std*sigma_multiplier, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.3),
            MetaBayesConv2d(in_channels=64, out_channels=128, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(128, device=device, sigma_init=std*sigma_multiplier, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MetaBayesConv2d(in_channels=128, out_channels=128, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(128, device=device, sigma_init=std*sigma_multiplier, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.4),
            MetaBayesConv2d(in_channels=128, out_channels=256, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(256, device=device, sigma_init=std*sigma_multiplier, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MetaBayesConv2d(in_channels=256, out_channels=256, kernel_size=3,
                            stride=1, padding="same", bias=bias, sigma_init=std*sigma_multiplier, device=device),
            MetaBayesBatchNorm2d(256, device=device, sigma_init=std*sigma_multiplier, track_running_stats=running_stats,
                                 affine=affine, eps=eps, momentum=momentum),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.5),
        )

        # data augmentation
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(),
        ])

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
        x = self.classifier(x)
        return x
    # add number of parameters total

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return super().extra_repr() + f"n_samples_train={self.n_samples_train}, sigma_init={self.std}, sigma_multiplier={self.sigma_multiplier}, version={self.version}, params={self.number_parameters()}"

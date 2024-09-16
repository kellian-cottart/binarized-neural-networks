from torch.nn import Module, BatchNorm2d, Conv2d, Sequential, ReLU, MaxPool2d, Dropout
from torchvision.transforms import v2
from .deepNeuralNetwork import *


class CifarNet(Module):
    """ CifarNet Neural Network 
    https://www.kaggle.com/code/farzadnekouei/cifar-10-image-classification-with-cnn#Step-4-%7C-Define-CNN-Model-Architecture
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
                 device: str = "cuda:0",
                 dropout: bool = False,
                 normalization: str = None,
                 bias: bool = False,
                 running_stats: bool = False,
                 affine: bool = False,
                 activation_function: str = "relu",
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            layers (list): List of layer sizes for the classifier
            device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
            dropout (bool): Whether to use dropout
            normalization (str): Normalization method
            bias (bool): Whether to use bias
            running_stats (bool): Whether to use running stats in BatchNorm
            affine (bool): Whether to use affine transformation in BatchNorm
            activation_function (str): Activation function
        """
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.normalization = normalization
        self.running_stats = running_stats
        self.affine = affine
        self.activation_function = activation_function
        self.bias = bias

        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(32, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(32, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(64, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(64, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.3),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(128, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(128, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.4),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(256, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                   stride=1, padding="same", bias=bias),
            BatchNorm2d(256, track_running_stats=running_stats,
                        affine=affine),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.5),
        )

        # data augmentation
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(),
        ])

        ## CLASSIFIER INITIALIZATION ##
        self.classifier = DNN(
            layers=layers,
            dropout=dropout,
            normalization=normalization,
            bias=bias,
            running_stats=running_stats,
            affine=affine,
            activation_function=activation_function,
            device=device,
            classifier=True,
            * args,
            **kwargs
        )

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        x = self.transform(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def number_parameters(self):
        """Return the number of parameters of the network"""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return super().extra_repr() + f"params={self.number_parameters()}"

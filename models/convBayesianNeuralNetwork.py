from .layers import *
from .convNeuralNetwork import ConvNN
from .bayesianNeuralNetwork import *
from typing import Union
from torch.nn import Dropout2d, GroupNorm, Identity, InstanceNorm2d, LayerNorm, MaxPool2d, functional


class ConvBayesianNeuralNetwork(ConvNN):
    """ Convolutional Binarized Neural Network(ConvBiNN)
    """

    def __init__(self,
                 layers: list = [1024, 1024, 10],
                 features: list = [64, 128, 256],
                 n_samples_train: int = 1,
                 zeroMean: bool = False,
                 sigma_multiplier: int = 1,
                 *args,
                 **kwargs):

        self.zeroMean = zeroMean
        self.n_samples_train = n_samples_train
        self.sigma_multiplier = sigma_multiplier
        super().__init__(layers=layers, features=features, *args, **kwargs)
        self.features = MetaBayesSequential(*self.features)
        self.classifier = BayesianNN(
            layers=layers, zeroMean=zeroMean, n_samples_train=n_samples_train, *args, **kwargs)

    def _features_init(self, features, bias=False):
        """ Initialize layers of the network for convolutional layers

            Args:
                features(list): List of layer sizes for the feature extractor
                bias(bool): Whether to use bias or not
        """
        # Add conv layers to the network as well as batchnorm and maxpool
        for i, _ in enumerate(features[:-1]):
            # Conv layers with BatchNorm and MaxPool
            self.features.append(MetaBayesConv2d(
                features[i], features[i+1], kernel_size=self.kernel_size[i], stride=self.stride, padding=self.padding, dilation=self.dilation, bias=bias, sigma_init=self.std*self.sigma_multiplier, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(MetaBayesConv2d(
                features[i+1], features[i+1], kernel_size=self.kernel_size[i], stride=self.stride, padding=self.padding, dilation=self.dilation, bias=bias, sigma_init=self.std*self.sigma_multiplier, device=self.device))
            self.features.append(self._norm_init(features[i+1]))
            self.features.append(self._activation_init())
            self.features.append(MaxPool2d(
                kernel_size=(2, 2)))
            if self.dropout == True:
                self.features.append(Dropout2d(p=0.2))

    def _weight_init(self, init='normal', std=0.1):
        pass

    def _norm_init(self, n_features):
        """Returns a layer of normalization"""
        if self.normalization == "batchnorm":
            return MetaBayesBatchNorm2d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats, sigma_init=self.std*self.sigma_multiplier).to(self.device)
        elif self.normalization == "layernorm":
            return LayerNorm(n_features).to(self.device)
        elif self.normalization == "instancenorm":
            return InstanceNorm2d(n_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.running_stats).to(self.device)
        elif self.normalization == "groupnorm":
            return GroupNorm(self.gnnum_groups, n_features).to(self.device)
        else:
            return Identity().to(self.device)

    def forward(self, x, *args, **kwargs):
        """Forward propagation of the binarized neural network"""
        x = self.features(x, self.n_samples_train)
        return self.classifier(x, self.n_samples_train)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", n_samples_train={self.n_samples_train}, zeroMean={self.zeroMean}, sigma_init={self.std}"

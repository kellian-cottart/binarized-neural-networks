from .layers import *
from .deepNeuralNetwork import *
from .layers.activation import *
from torch.nn import Dropout, LayerNorm, InstanceNorm1d, GroupNorm, Identity


class BayesianNN(DNN):
    """ Neural Network Base Class
    """

    def __init__(self,
                 layers,
                 zeroMean=False,
                 n_samples_train=1,
                 *args,
                 **kwargs):
        """ NN initialization

        Args:
            n_samples_train (int): Number of forward samples
            n_samples_backward (int): Number of backward samples
        """
        self.zeroMean = zeroMean
        self.n_samples_train = n_samples_train
        if "classifier" in kwargs:
            self.classifier = kwargs["classifier"]
        super().__init__(layers, *args, **kwargs)

    def _layer_init(self, layers, bias=False):
        """ Initialize layers of NN

        Args:
            dropout (bool): Whether to use dropout
            bias (bool): Whether to use bias
        """
        self.layers = MetaBayesSequential(*self.layers)
        self.layers.append(nn.Flatten(start_dim=1).to(self.device))
        for i, _ in enumerate(layers[:-1]):
            self.layers.append(MetaBayesLinearParallel(
                in_features=layers[i],
                out_features=layers[i+1],
                bias=bias,
                zeroMean=self.zeroMean,
                sigma_init=self.std,
                device=self.device
            ))
            self.layers.append(self._norm_init(layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(self._activation_init())
            if self.dropout and i < len(layers)-2:
                self.layers.append(Dropout(p=0.5))

    def _weight_init(self, init='normal', std=0.1):
        pass

    def _norm_init(self, n_features):
        """
        Args:
            n_features (int): Number of features

        Returns:
            Module: Normalization layer module
        """
        normalization_layers = {
            "batchnorm": lambda: MetaBayesBatchNorm1d(
                n_features,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.running_stats,
                sigma_init=self.std),
            "layernorm": lambda: LayerNorm(n_features),
            "instancenorm": lambda: InstanceNorm1d(n_features, eps=self.eps, affine=self.affine, track_running_stats=self.running_stats),
            "groupnorm": lambda: GroupNorm(self.gnnum_groups, n_features),
        }
        return normalization_layers.get(self.normalization, Identity)().to(self.device)

    def forward(self, x, *args, **kwargs):
        """ Forward pass of DNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        repeat_samples = self.n_samples_train if self.n_samples_train > 1 else 1
        samples = self.n_samples_train
        if x.dim() == 4 and not hasattr(self, "classifier"):
            x = x.repeat(samples, *(1,)*len(x.size()[1:]))
        out = self.layers(x, samples)
        return out.reshape(repeat_samples, out.size(0)//repeat_samples, *out.size()[1:])

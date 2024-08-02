import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union


class GaussianParameter:
    """Object used to perform the reparametrization tricks in gaussian sampling and reshape the tensor of samples in the right shape to prevents a for loop over the number of sample"""

    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu  # Mean of the distribution
        self.sigma = sigma  # Standard deviation of the distribution

    def sample(self, samples=1):
        """Sample from the Gaussian distribution using the reparameterization trick."""
        # Sample from the standard normal and adjust with sigma and mu
        if samples == 0:
            return self.mu.unsqueeze(0)
        buffer_epsilon = self.sigma.unsqueeze(0).repeat(
            samples, *([1]*len(self.sigma.shape)))
        epsilon = torch.empty_like(buffer_epsilon).normal_()
        mu = self.mu.unsqueeze(0).repeat(
            samples, *([1]*len(self.mu.shape)))
        return mu + buffer_epsilon * epsilon


class MetaBayesConvNd(Module):
    """As in pytorch, exceptweights and biases are gaussian, each samples are a group."""

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:

        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    samples: int  # replace groups in F.conv2d, weights samples are concatenated across the first dimension
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    sigma_init: float

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 sigma_init: float,
                 bias: bool,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MetaBayesConvNd, self).__init__()

        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        self.sigma_init = sigma_init

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2)

        if transposed == True:
            self.weight_sigma = Parameter(torch.empty(
                (in_channels, out_channels, *kernel_size), **factory_kwargs))
            self.weight_mu = Parameter(torch.empty(
                (in_channels, out_channels, *kernel_size), **factory_kwargs))

        else:
            self.weight_sigma = Parameter(torch.empty(
                (out_channels, in_channels, *kernel_size), **factory_kwargs))
            self.weight_mu = Parameter(torch.empty(
                (out_channels, in_channels, *kernel_size), **factory_kwargs))

        self.weight = GaussianParameter(self.weight_mu, self.weight_sigma)
        if bias == True:
            self.bias_sigma = Parameter(
                torch.empty(out_channels, **factory_kwargs))
            self.bias_mu = Parameter(torch.empty(
                out_channels, **factory_kwargs))
            self.bias = GaussianParameter(self.bias_mu, self.bias_sigma)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight_mu)
        self.bound = math.sqrt(6/(fan_in+fan_out))
        init.uniform_(self.weight_mu, -self.bound, self.bound)
        init.constant_(self.weight_sigma, self.sigma_init)
        if self.bias is not None:
            if fan_in != 0:
                init.constant_(self.bias_mu, 0)
                init.constant_(self.bias_sigma, self.sigma_init)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, sigma_init={sigma_init}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(MetaBayesConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class MetaBayesConv2d(MetaBayesConvNd):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            sigma_init: float = 0.001 ** 0.5,
            bias: bool = True,
            padding_mode: str = 'zeros',  # TODO: refine this type
            device=None,
            dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(MetaBayesConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            transposed=False, output_padding=_pair(0), sigma_init=sigma_init, bias=bias, padding_mode=padding_mode, **factory_kwargs)

    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor], samples=1):
        """ To compute the monte carlo sampling in an optimum manner, we use the functionality group of F.conv2d """
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, groups=samples)
        return F.conv2d(x, weight, bias, self.stride,
                        self.padding, self.dilation, groups=samples)

    def forward(self, x: Tensor, samples=1) -> Tensor:
        """ The shapings are necessary to take into account that:
            - feature map are made of monte carlo samples 
            - the input of the neural network is not a monte carlo sample"""
        W = self.weight.sample(samples)
        W = W.view(W.size(0)*W.size(1), W.size(2), W.size(3), W.size(4))
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            x = x.repeat(samples, 1, 1, 1, 1)
        x = x.view(x.size(1), x.size(0)*x.size(2), x.size(3), x.size(4))
        B = self.bias.sample(samples).flatten(
        ) if self.bias is not None else None
        samples = samples if samples > 0 else 1
        out = self._conv_forward(x, W, B, samples)
        return out.view(samples, out.size(0), out.size(1) // samples, out.size(2), out.size(3))

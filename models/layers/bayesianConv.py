from math import sqrt
from torch import Tensor, stack
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
from .gaussianParameter import *


class MetaBayesConvNd(Module):
    """As in pytorch, exceptweights and biases are gaussian, each samples are a group."""

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[Tensor]}

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
                 groups: int,
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
        self.groups = groups

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

        ## WEIGHTS INITIALIZATION ##
        self.weight = GaussianParameter(
            out_features=out_channels,
            in_features=in_channels // groups,
            kernel_size=kernel_size,
            **factory_kwargs)
        if bias == True:
            self.bias = GaussianParameter(
                out_features=out_channels,
                **factory_kwargs)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight.mu)
        self.bound = sqrt(6/(fan_in+fan_out))
        init.uniform_(self.weight.mu, -self.bound, self.bound)
        init.constant_(self.weight.sigma, self.sigma_init)
        if self.bias is not None:
            if fan_in != 0:
                init.constant_(self.bias.mu, 0)
                init.constant_(self.bias.sigma, self.sigma_init)

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
        if self.groups != 1:
            s += ', groups={groups}'
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
            groups: int = 1,
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
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, groups=groups,
            transposed=False, output_padding=_pair(0), sigma_init=sigma_init, bias=bias, padding_mode=padding_mode, **factory_kwargs)

    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor], samples=1) -> Tensor:
        """ To compute the monte carlo sampling in an optimum manner, we use the functionality group of F.conv2d """
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, groups=self.groups*samples)
        return F.conv2d(
            x=x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups*samples)

    def forward(self, x: Tensor, samples) -> Tensor:
        """ The shapings are necessary to take into account that:
            - feature map are made of monte carlo samples
            - the input of the neural network is not a monte carlo sample"""
        act_mu = F.conv2d(x, self.weight.mu, self.bias.mu if self.bias is not None else None, self.stride,
                          self.padding, self.dilation, self.groups)
        act_sigma = F.conv2d(x**2, self.weight.sigma**2, self.bias.sigma**2 if self.bias is not None else None, self.stride,
                             self.padding, self.dilation, self.groups)
        samples = samples if samples > 1 else 1
        epsilon = empty_like(act_mu).normal_()
        out = act_mu + act_sigma**0.5 * epsilon
        return out

#   def forward(self, x: Tensor, samples) -> Tensor:
#         """ The shapings are necessary to take into account that:
#             - feature map are made of monte carlo samples
#             - the input of the neural network is not a monte carlo sample"""
#         weights = self.weight.sample(samples)
#         B = self.bias.sample(samples).flatten(
#         ) if self.bias is not None else None
#         samples = samples if samples > 1 else 1
#         weights = weights.reshape(weights.size(
#             0)*weights.size(1), *weights.size()[2:])


#         x = x.reshape(x.size(0)//samples, samples*x.size(1), *x.size()[2:])
#         out = self._conv_forward(x, weights, B, samples)
#         out = out.reshape(out.size(0)*samples, out.size(1) //
#                           samples, *out.size()[2:])
#         return out

from torch import Tensor, zeros, ones, tensor, long, stack
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from typing import Optional
from .gaussianParameter import *


class MetaBayesNorm(Module):
    """Common base of _InstanceNorm and _BatchNorm.
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py
    """

    _version = 1
    __constants__ = ["track_running_stats",
                     "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        sigma_init: float = 0.1,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.sigma_init = sigma_init
        if self.affine:
            self.weight = GaussianParameter(num_features, **factory_kwargs)
            self.bias = GaussianParameter(num_features, **factory_kwargs)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", zeros(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_var", ones(num_features, **factory_kwargs)
            )
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                "num_batches_tracked",
                tensor(
                    0,
                    dtype=long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            # type: ignore[union-attr,operator]
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight.mu)
            init.constant_(self.weight.sigma, self.sigma_init)
            init.zeros_(self.bias.mu)
            init.constant_(self.bias.sigma, self.sigma_init)

    def _check_input_dim(self, x):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}, sigma_init={sigma_init}".format(
                **self.__dict__)
        )


class MetaBayesBatchNorm(MetaBayesNorm):
    """https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        sigma_init: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features=num_features,
            eps=eps,
            sigma_init=sigma_init,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            **factory_kwargs,
        )

    def forward(self, x: Tensor, samples: int) -> Tensor:
        self._check_input_dim(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if self.affine == True:
            # Sample the weights
            weights = self.weight.sample(samples)
            biases = self.bias.sample(samples)
            weights = weights.reshape(weights.size(
                0)*weights.size(1), *weights.size()[2:])
            biases = biases.reshape(biases.size(
                0)*biases.size(1), *biases.size()[2:])
        samples = samples if samples > 1 else 1
        x = x.reshape(x.size(0)//samples,
                      samples*x.size(1), *x.size()[2:])
        if self.track_running_stats or (not self.training and self.track_running_stats):
            running_mean = self.running_mean.repeat(
                samples, *([1] * (len(self.running_mean.size())-1)))
            running_var = self.running_var.repeat(
                samples, *([1] * (len(self.running_var.size())-1)))
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                weights if self.affine else None,
                biases if self.affine else None,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
            self.running_mean = running_mean.reshape(samples, running_mean.size(
                0)//samples, *running_mean.size()[1:]).mean(dim=0)
            self.running_var = running_var.reshape(samples, running_var.size(
                0)//samples, *running_var.size()[1:]).mean(dim=0)
        else:
            out = F.batch_norm(
                x,
                None,
                None,
                weights if self.affine else None,
                biases if self.affine else None,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        return out.reshape(out.size(0)*samples, out.size(1)//samples, *out.size()[2:])


class MetaBayesBatchNorm1d(MetaBayesBatchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(
                    x.dim())
            )


class MetaBayesBatchNorm2d(MetaBayesBatchNorm):
    def _check_input_dim(self, x):
        if x.dim() != 3 and x.dim() != 4:
            raise ValueError(
                "expected 3D input or 4D input (got {}D input)".format(
                    x.dim())
            )

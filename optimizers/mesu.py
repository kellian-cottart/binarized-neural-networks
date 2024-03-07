import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Optional
from copy import deepcopy

__all__ = ['MESU', 'mesu', 'BGD', 'bgd']


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad


class MESU(Optimizer):
    r"""
    Implements the Meta Bayes algorithm (optionally with momentum).

    Algorithm in Latex:
    ...
    (Your algorithm description goes here)
    ...

    Args:
        params: Model parameters.
        coeff_likeli_mu: Coefficient for likelihood term.
        coeff_likeli_sigma: Coefficient for likelihood term.
        sigma_p: Prior sigma value.
        sigma_b: Beta sigma value.
        alpha: Alpha coefficient.
        update: Update method (1, 2, 3).
        keep_prior: Whether to keep the prior values.
        differentiable: Whether the optimizer is differentiable.

    Raises:
        ValueError: If input arguments are invalid.

    Note:
        Make sure your module has `sigma` defined before `mu`.

    """

    def __init__(self, params, coeff_likeli_mu=1, coeff_likeli_sigma=1, update=1, sigma_p=0.06, sigma_b=10, alpha=0.01, keep_prior=False, differentiable=False, clamp_grad=False):
        # Validation of input arguments
        if coeff_likeli_mu < 0.0:
            raise ValueError(
                f"Invalid likelihood coefficient: {coeff_likeli_mu}")
        if coeff_likeli_sigma < 0.0:
            raise ValueError(
                f"Invalid likelihood coefficient: {coeff_likeli_sigma}")
        if sigma_p < 0.0:
            raise ValueError(f"Invalid sigma prior value: {sigma_p}")
        if sigma_b < 0.0:
            raise ValueError(f"Invalid sigma beta value: {sigma_b}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha coefficient: {alpha}")
        if update not in [1, 2, 3]:
            raise ValueError(f"Invalid update method: {update}")

        defaults = dict(coeff_likeli_mu=coeff_likeli_mu, coeff_likeli_sigma=coeff_likeli_sigma,
                        sigma_p=sigma_p, sigma_b=sigma_b, alpha=alpha, update=update,
                        keep_prior=keep_prior, differentiable=differentiable, clamp_grad=clamp_grad)

        print("Careful! This optimizer takes only Meta Bayes parameters!")
        print("In your Module, sigma must be defined before mu")

        super().__init__(params, defaults)

        num_params = 0
        for group in self.param_groups:
            num_params += len(group['params'])
            if group['keep_prior']:
                group.setdefault('priors', deepcopy(group['params']))

        print(f'You have {num_params} Tensors of Meta Bayes parameters')

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('keep_prior', False)
            group.setdefault('differentiable', False)
            group.setdefault('clamp_grad', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            if group['keep_prior']:
                priors_with_grad = []
            else:
                priors_with_grad = None
            d_p_list = []
            has_sparse_grad = False

            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    if group['keep_prior']:
                        priors_with_grad.append(group['priors'][i])
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    state = self.state[p]

            mesu(params_with_grad,
                 d_p_list,
                 priors_with_grad,
                 coeff_likeli_mu=group['coeff_likeli_mu'],
                 coeff_likeli_sigma=group['coeff_likeli_sigma'],
                 sigma_p=group['sigma_p'],
                 sigma_b=group['sigma_b'],
                 alpha=group['alpha'],
                 update=group['update'],
                 keep_prior=group['keep_prior'],
                 clamp_grad=group['clamp_grad'],
                 has_sparse_grad=has_sparse_grad)

        return loss


def mesu(params: List[Tensor],
         d_p_list: List[Tensor],
         prior: List[Optional[Tensor]],
         has_sparse_grad: bool = None,
         *,
         coeff_likeli_mu: float,
         coeff_likeli_sigma: float,
         sigma_p: float,
         sigma_b: float,
         alpha: float,
         update: int,
         keep_prior: bool,
         clamp_grad: bool):

    if not params:
        raise ValueError('Your parameters have no gradients!')

    if len(params) % 2 == 1:
        raise ValueError(
            'You must have Sigma and Mu in the same parameter group')

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        is_sigma = not (i % 2)
        if clamp_grad:
            d_p = torch.clamp(d_p, -0.1, 0.1)
        if is_sigma:
            V = param.data ** 2
            if update == 1:
                param.data.add_(-coeff_likeli_sigma * V * d_p)
            elif update == 2:
                param.data.add_(-coeff_likeli_sigma * V * d_p)
                param.data.mul_(1 / (1 - (alpha ** 2)))
            elif update == 3:
                S = param.data
                if keep_prior:
                    param.data.add_(-coeff_likeli_sigma * V * d_p +
                                    S * (prior[i] ** 2 - V) / (sigma_b ** 2))
                else:
                    param.data.add_(-coeff_likeli_sigma * V *
                                    d_p + S * (sigma_p ** 2 - V) / (sigma_b ** 2))
        else:
            if keep_prior:
                param.data.add_(-coeff_likeli_mu * V * d_p + 0.5 *
                                S * (prior[i] - param.data) / (sigma_b ** 2))
            else:
                param.data.add_(-coeff_likeli_mu * V * d_p)


class BGD(Optimizer):
    r"""
    Implements the Meta Bayes algorithm (optionally with momentum).

    Algorithm in Latex:
    ...
    (Your algorithm description goes here)
    ...

    Args:
        params: Model parameters.
        coeff_likeli_mu: Coefficient for likelihood term.
        differentiable: Whether the optimizer is differentiable.

    Raises:
        ValueError: If input arguments are invalid.

    Note:
        Make sure your module has `sigma` defined before `mu`.

    """

    def __init__(self, params, coeff_likeli_mu=1, differentiable=False):
        # Validation of input arguments
        if coeff_likeli_mu < 0.0:
            raise ValueError(
                f"Invalid likelihood coefficient: {coeff_likeli_mu}")

        defaults = dict(coeff_likeli_mu=coeff_likeli_mu,
                        differentiable=differentiable)

        print("Careful! This optimizer takes only Meta Bayes parameters!")
        print("In your Module, sigma must be defined before mu")

        super(BGD, self).__init__(params, defaults)

        num_params = 0
        for group in self.param_groups:
            num_params += len(group['params'])

        print(f'You have {num_params} Tensors of Meta Bayes parameters')

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    state = self.state[p]

            bgd(params_with_grad,
                d_p_list,
                coeff_likeli_mu=group['coeff_likeli_mu'],
                has_sparse_grad=has_sparse_grad)

        return loss


def bgd(params: List[Tensor], d_p_list: List[Tensor],  coeff_likeli_mu: float, has_sparse_grad: bool = None):
    if not params:
        raise ValueError('Your parameters have no gradients!')

    if len(params) == 1:
        raise ValueError(
            'You must have Sigma and Mu in the same parameter group')

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        is_sigma = not (i % 2)

        if is_sigma:
            V = param.data ** 2
            param.data.add_(-0.5 * V * d_p - param.data *
                            (-1 + (1 + 0.25 * (V * (d_p ** 2))) ** 0.5))
        else:
            param.data.add_(-coeff_likeli_mu * V * d_p)

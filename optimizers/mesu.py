#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:10:54 2024

@author: Dr Djo ;)
"""

from torch import Tensor, sqrt, norm, prod, tensor, trace
from torch.optim.optimizer import Optimizer
from typing import List

__all__ = ['MESU', 'mesu']


class MESU(Optimizer):
    r"""
    Implements Metaplasticity from Synaptic Uncertainty without alterations.

    Args:
        params: Model parameters. Parameters representing 'sigma' must be defined before 'mu'.
        sigma_prior: Standard deviation of the prior over the weights, typically 0.01**0.5, or 0.001**0.5 . Can vary per group.
        N: Number of batches to retain in synaptic memory in the Bayesian Forgetting framework. N acts as a weight between prior and likelihood. It should be DATASET_SIZE//BATCH_SIZE, but using larger values can yield better results.
        clamp_grad: If >0, clamps the gradient of the loss over mu and sigma to this value, typically 0.1 or 1.

    Raises:
        ValueError: If input arguments are invalid.

    Note:
        Additional parameters can accelerate learning, such as:
            - Extra learning rates for mu and sigma
            - Individual priors for each synapse
    """

    def __init__(self, params, lr_sigma=1, lr_mu=1, sigma_prior=0.1, mu_prior=0.1, N_mu=1e5, N_sigma=1e5, norm_term=False):

        if sigma_prior <= 0:
            raise ValueError(
                f'sigma_prior must be positive, got {sigma_prior}')
        defaults = dict(
            sigma_prior=sigma_prior,
            mu_prior=mu_prior,
            N_mu=N_mu,
            N_sigma=N_sigma,
            lr_sigma=lr_sigma,
            lr_mu=lr_mu,
            norm_term=norm_term,
        )

        super().__init__(params, defaults)

        num_params = sum(len(group['params']) for group in self.param_groups)
        print(
            f'Optimizer initialized with {num_params} Gaussian variational parameters.')

    def step(self,):
        """Performs a single optimization step."""
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                # print p name
                if p.grad is None:
                    continue
                d_p_list.append(p.grad)
                params_with_grad.append(p)

            mesu(
                params_with_grad,
                d_p_list,
                sigma_prior=group['sigma_prior'],
                mu_prior=group['mu_prior'],
                N_mu=group['N_mu'],
                N_sigma=group['N_sigma'],
                lr_sigma=group['lr_sigma'],
                lr_mu=group['lr_mu'],
                norm_term=group['norm_term'],
            )


def mesu(params: List[Tensor], d_p_list: List[Tensor], sigma_prior: float, mu_prior: float, N_mu: int, N_sigma: float, lr_mu: float, lr_sigma: float, norm_term: bool = False):
    if not params:
        raise ValueError('No gradients found in parameters!')
    if len(params) % 2 == 1:
        raise ValueError(
            'Parameters must include both Sigma and Mu in each group.')
    for sigma, mu, grad_sigma, grad_mu in zip(params[::2], params[1::2], d_p_list[::2], d_p_list[1::2]):
        variance = sigma.data ** 2
        forgetting_mu = N_mu * (sigma_prior ** 2)
        forgetting_sigma = N_sigma * (sigma_prior ** 2)
        square_root = 1 / ((grad_sigma.abs()/sigma).mean()) if norm_term else 1
        second_order_mu = 1 + variance * \
            ((grad_mu**2) - (1/(N_mu*variance) - 1/(N_mu*(sigma_prior**2))))
        second_order_sigma = 1 + variance * \
            ((grad_mu**2) - (1/(N_sigma*variance) -
             1/(N_sigma*(sigma_prior**2))))

        mu.data = mu.data + (- lr_mu * variance * grad_mu + variance *
                             (mu_prior - mu.data) / forgetting_mu) / second_order_mu
        sigma.data = sigma.data + (- 0.5 * lr_sigma * variance * grad_sigma * square_root +
                                   0.5 * sigma * (sigma_prior ** 2 - variance) / forgetting_sigma) / second_order_sigma

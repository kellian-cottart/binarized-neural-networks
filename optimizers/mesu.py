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

    def __init__(self, params, lr_sigma=1, lr_mu=1, sigma_prior=0.1, mu_prior=0.1, N_mu=1e5, N_sigma=1e5, clamp_grad=0):

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
            clamp_grad=clamp_grad,
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
                clamp_grad=group['clamp_grad'],
            )


def mesu(params: List[Tensor], d_p_list: List[Tensor], sigma_prior: float, mu_prior: float, N_mu: int, N_sigma: float, lr_mu: float, lr_sigma: float, clamp_grad: float = 0):
    if not params:
        raise ValueError('No gradients found in parameters!')
    if len(params) % 2 == 1:
        raise ValueError(
            'Parameters must include both Sigma and Mu in each group.')
    for sigma, mu, grad_sigma, grad_mu in zip(params[::2], params[1::2], d_p_list[::2], d_p_list[1::2]):
        grad_mu.data = grad_mu.data * lr_mu
        grad_sigma.data = grad_sigma.data * lr_sigma
        if clamp_grad > 0:
            grad_mu.data.clamp_(min=-clamp_grad, max=clamp_grad)
            grad_sigma.data.clamp_(min=-clamp_grad, max=clamp_grad)
        variance = sigma.data ** 2
        prior_attraction_mu = variance * \
            (mu_prior - mu.data) / N_mu * (sigma_prior ** 2)
        prior_attraction_sigma = 0.5 * sigma * \
            (sigma_prior ** 2 - variance) / (N_sigma * (sigma_prior ** 2))
        mu.data = mu.data + (-variance * grad_mu +
                             prior_attraction_mu)
        sigma.data = sigma.data + \
            (- 0.5 * variance * grad_sigma +
             prior_attraction_sigma)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:10:54 2024

@author: Dr Djo ;)
"""

from torch import Tensor
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

    def __init__(self, params, lr=1, sigma_prior=0.1, N=1e5, sigma_grad_divide=1):

        if sigma_prior <= 0:
            raise ValueError(
                f'sigma_prior must be positive, got {sigma_prior}')
        if N <= 0:
            raise ValueError(f'N must be positive, got {N}')

        defaults = dict(
            sigma_prior=sigma_prior,
            N=N,
            lr=lr,
            sigma_grad_divide=sigma_grad_divide,
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
                N=group['N'],
                lr=group['lr'],
                sigma_grad_divide=group['sigma_grad_divide'],
            )


def mesu(params: List[Tensor], d_p_list: List[Tensor], sigma_prior: float, N: int, lr: float, sigma_grad_divide: float):
    if not params:
        raise ValueError('No gradients found in parameters!')
    if len(params) % 2 == 1:
        raise ValueError(
            'Parameters must include both Sigma and Mu in each group.')
    for sigma, mu, grad_sigma, grad_mu in zip(params[::2], params[1::2], d_p_list[::2], d_p_list[1::2]):
        variance = sigma.data ** 2
        forgetting = N * (sigma_prior ** 2)
        second_order_mu = (N - 1)/N + variance * \
            (grad_mu ** 2) + variance/forgetting
        second_order_sigma = (N - 1)/N + sigma_grad_divide*variance * \
            (grad_sigma ** 2) + variance/forgetting
        mu.data = mu.data - lr * variance * grad_mu / \
            second_order_mu - variance * mu.data / \
            (forgetting * second_order_mu)
        sigma.data = sigma.data - 0.5 * variance * grad_sigma / \
            second_order_sigma + 0.5 * sigma.data * \
            (sigma_prior ** 2 - variance) / (forgetting * second_order_sigma)

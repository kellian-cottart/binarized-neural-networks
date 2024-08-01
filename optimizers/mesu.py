#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:10:54 2024

@author: Dr Djo ;) 
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional

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

    def __init__(self, params, sigma_prior=0.1, N=1e5, clamp_grad=0):

        if sigma_prior <= 0:
            raise ValueError(
                f'sigma_prior must be positive, got {sigma_prior}')
        if N <= 0:
            raise ValueError(f'N must be positive, got {N}')

        defaults = dict(
            sigma_prior=sigma_prior,
            N=N,
            clamp_grad=clamp_grad
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

            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    d_p_list.append(p.grad)
                    params_with_grad.append(p)

            mesu(
                params_with_grad,
                d_p_list,
                sigma_prior=group['sigma_prior'],
                N=group['N'],
                clamp_grad=group['clamp_grad'],
            )


def mesu(params: List[Tensor], d_p_list: List[Tensor], sigma_prior: float, N: int, clamp_grad: float):
    if not params:
        raise ValueError('No gradients found in parameters!')
    if len(params) % 2 == 1:
        raise ValueError(
            'Parameters must include both Sigma and Mu in each group.')

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if clamp_grad > 0:
            d_p = torch.clamp(d_p, -clamp_grad, clamp_grad)

        is_sigma = (i % 2 == 0)
        if is_sigma:
            variance = param.data ** 2
            param.data.add_(-0.5 * variance * d_p + param.data *
                            (sigma_prior ** 2 - variance) / (N * sigma_prior ** 2))
        else:
            param.data.add_(-variance * d_p - variance *
                            param.data / (N * sigma_prior ** 2))

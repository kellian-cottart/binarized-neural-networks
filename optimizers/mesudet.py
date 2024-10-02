#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:10:54 2024

@author: Dr Djo ;) 
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple
from torch.nn import Module
import torch.nn.init as init
import copy


class MESUDET(object):
    r"""
    Implements Metaplasticity from Synaptic Uncertainty without alterations.

    Args:
        model: The entire Model parameters. We iterate over the models params that may or may not have gradients. 
        sigma_prior: Standard deviation of the prior over the weights, typically 0.01**0.5, or 0.001**0.5 . Can vary per group.
        N: Number of batches to retain in synaptic memory in the Bayesian Forgetting framework. N acts as a weight between prior and likelihood. It should be DATASET_SIZE//BATCH_SIZE, but using larger values can yield better results.
        clamp_sigma: If >0, clamps the  sigma to this value times sigma_prior, typically 1e-3 or 1e-2.

    Raises:
        ValueError: If input arguments are invalid.

    Note:
        Additional parameters can accelerate learning, such as:
            - Extra learning rates for mu and sigma
            - Individual priors for each synapse
    """

    def __init__(self, model, **args_dict):

        super().__init__()
        self.model = model
        self.mu_prior = args_dict['mu_prior']
        self.sigma_prior = args_dict['sigma_prior']
        self.N_mu = args_dict['N_mu']
        self.N_sigma = args_dict['N_sigma']
        self.c_sigma = args_dict['c_sigma']
        self.c_mu = args_dict['c_mu']
        self.second_order = args_dict['second_order']
        self.clamp_sigma = args_dict['clamp_sigma']
        self.clamp_mu = args_dict['clamp_mu']
        self.enforce_learning_sigma = args_dict['enforce_learning_sigma']
        num_params = len(list(model.parameters()))
        print(
            f'Optimizer initialized with {num_params} Gaussian variational Tensor parameters.')

    def step(self,):
        """Performs a single optimization step."""
        mesu(model=self.model,
             mu_prior=self.mu_prior,
             sigma_prior=self.sigma_prior,
             N_mu=self.N_mu,
             N_sigma=self.N_sigma,
             c_sigma=self.c_sigma,
             c_mu=self.c_mu,
             second_order=self.second_order,
             clamp_sigma=self.clamp_sigma,
             clamp_mu=self.clamp_mu,
             enforce_learning_sigma=self.enforce_learning_sigma
             )

    def zero_grad(self):
        """Sets the gradients of all model parameters to zero."""
        self.model.zero_grad()


def mesu(model: Module, *, mu_prior: float, sigma_prior: float, N_mu: int, N_sigma: int, c_sigma: float, c_mu: float, second_order: bool, clamp_sigma: list, clamp_mu: list, enforce_learning_sigma: int):

    previous_param = None
    for i, (name, param) in enumerate(model.named_parameters(recurse=True)):
        if previous_param is None:
            sigma = param
            variance = param.data**2
            grad_sigma = param.grad
            previous_param = 'sigma'
        else:
            mu = param
            grad_mu = param.grad
            previous_param = None
            if grad_sigma != None and grad_mu != None:
                grad_sigma = c_sigma*grad_sigma
                grad_mu = c_mu*grad_mu
                denominator_sigma = 1 + (variance *
                                         ((grad_mu**2) - (1/(N_sigma*variance) -
                                                          1/(N_sigma*(sigma_prior**2)))))*second_order
                denominator_mu = 1 + (variance *
                                      ((grad_mu**2) - (1/(N_mu*variance) - 1/(N_mu*(sigma_prior**2)))))*second_order
                prior_attraction_mu = variance * \
                    (mu_prior - mu.data) / N_mu * (sigma_prior ** 2)
                prior_attraction_sigma = 0.5 * sigma * \
                    (sigma_prior ** 2 - variance) / \
                    (N_sigma * (sigma_prior ** 2))
                mu.data = mu.data + (-variance * grad_mu +
                                     prior_attraction_mu) / denominator_mu
                sigma.data = sigma.data + \
                    (- 0.5 * variance * grad_sigma +
                     prior_attraction_sigma) / denominator_sigma
                if clamp_sigma[0] != 0:
                    sigma.data = torch.clamp(
                        sigma.data, clamp_sigma[0], clamp_sigma[1])
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(mu.data, clamp_mu[0], clamp_mu[1])
            if grad_sigma == None and grad_mu != None:
                denominator_mu = 1 + (variance *
                                      ((grad_mu**2) - (1/(N_mu*variance) - 1/(N_mu*(sigma_prior**2)))))*second_order
                prior_attraction_mu = variance * \
                    (mu_prior - mu.data) / N_mu * (sigma_prior ** 2)
                mu.data = mu.data + (-variance * grad_mu +
                                     prior_attraction_mu) / denominator_mu
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(mu.data, clamp_mu[0], clamp_mu[1])
            if grad_sigma == None and enforce_learning_sigma == True and grad_mu != None:
                grad_sigma = sigma * (grad_mu**2) / (grad_mu**2).mean()
                grad_sigma = c_sigma*grad_sigma
                denominator_sigma = 1 + (variance *
                                         ((grad_mu**2) - (1/(N_sigma*variance) -
                                                          1/(N_sigma*(sigma_prior**2)))))*second_order
                prior_attraction_sigma = 0.5 * sigma * \
                    (sigma_prior ** 2 - variance) / \
                    (N_sigma * (sigma_prior ** 2))
                sigma.data = sigma.data + \
                    (- 0.5 * variance * grad_sigma +
                     prior_attraction_sigma) / denominator_sigma
                if clamp_sigma[0] != 0:
                    sigma.data = torch.clamp(
                        sigma.data, clamp_sigma[0], clamp_sigma[1])


class BGDDET(object):
    r"""
    Implements Bayesian Gradient Descent based on the work of Chen Zeno --> https://arxiv.org/pdf/1803.10123.

    Args:
        params: Model parameters. Parameters representing 'sigma' must be defined before 'mu'.
        learning_rate: It should be 1. Can be greater to adjust the convergence rate. 

    Raises:
        ValueError: If input arguments are invalid.
    """

    def __init__(self, model, args_dict):

        super().__init__()
        self.model = model
        self.normalise_grad_sigma = args_dict['normalise_grad_sigma']
        self.normalise_grad_mu = args_dict['normalise_grad_mu']
        self.c_sigma = args_dict['c_sigma']
        self.c_mu = args_dict['c_mu']
        self.second_order = args_dict['second_order']
        self.clamp_sigma = args_dict['clamp_sigma']
        self.clamp_mu = args_dict['clamp_mu']
        num_params = len(list(model.parameters()))
        print(
            f'Optimizer initialized with {num_params} Gaussian variational Tensor parameters.')

    def step(self):
        """Performs a single optimization step."""

        bgd(model=self.model,
            normalise_grad_sigma=self.normalise_grad_sigma,
            normalise_grad_mu=self.normalise_grad_mu,
            c_sigma=self.c_sigma,
            c_mu=self.c_mu,
            second_order=self.second_order,
            clamp_sigma=self.clamp_sigma,
            clamp_mu=self.clamp_mu
            )


def bgd(model: Module, *, normalise_grad_sigma: int, normalise_grad_mu: int, c_sigma: float, c_mu: float, second_order: bool, clamp_sigma: list, clamp_mu: list):
    previous_param = None
    for i, (name, param) in enumerate(model.named_parameters(recurse=True)):

        if previous_param is None:
            sigma = param
            variance = param.data**2
            grad_sigma = param.grad
            previous_param = 'sigma'

        else:
            mu = param
            grad_mu = param.grad
            previous_param = None
            if grad_sigma != None and grad_mu != None:
                square_root_sigma = 1e-12 + (normalise_grad_sigma == 0)*1 + (normalise_grad_sigma == 1)*(grad_sigma**2).mean()**0.5 + (
                    normalise_grad_sigma == 2)*(grad_mu**2).mean()**0.5 + (normalise_grad_sigma == 3)*((grad_sigma**2).mean()**0.5+(grad_mu**2).mean()**0.5)
                square_root_mu = 1e-12 + (normalise_grad_mu == 0)*1 + (normalise_grad_mu == 1)*(grad_sigma**2).mean()**0.5 + (
                    normalise_grad_mu == 2)*(grad_mu**2).mean()**0.5 + (normalise_grad_mu == 3)*((grad_sigma**2).mean()**0.5+(grad_mu**2).mean()**0.5)
                grad_sigma = c_sigma*grad_sigma/square_root_sigma
                grad_mu = c_mu*grad_mu/square_root_mu
                denominator_sigma = 1 + second_order * \
                    (variance * grad_sigma ** 2)
                denominator_mu = 1 + second_order * (variance * grad_mu ** 2)
                sigma.data.add_(-(0.5 * variance * grad_sigma - sigma.data * (-1 + (
                    1 + 0.25 * (variance * (grad_sigma ** 2))) ** 0.5)) / denominator_sigma)
                mu.data.add_(-(variance * grad_mu) / denominator_mu)
                if clamp_sigma[0] != 0:
                    sigma.data = torch.clamp(
                        sigma.data, clamp_sigma[0], clamp_sigma[1])
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(sigma.data, clamp_mu[0], clamp_mu[1])

            if grad_sigma == None and grad_mu != None:
                denominator_mu = 1 + second_order*variance * grad_mu ** 2
                mu.data.add_(-(variance * grad_mu) / denominator_mu)
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(sigma.data, clamp_mu[0], clamp_mu[1])
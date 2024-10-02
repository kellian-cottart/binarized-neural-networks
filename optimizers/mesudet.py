#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:10:54 2024

@author: Dr Djo ;)
"""

import torch
from torch.nn import Module


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
        self.normalise_grad_sigma = args_dict['normalise_grad_sigma']
        self.normalise_grad_mu = args_dict['normalise_grad_mu']
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
             normalise_grad_sigma=self.normalise_grad_sigma,
             normalise_grad_mu=self.normalise_grad_mu,
             c_sigma=self.c_sigma,
             c_mu=self.c_mu,
             second_order=self.second_order,
             clamp_sigma=self.clamp_sigma,
             clamp_mu=self.clamp_mu,
             enforce_learning_sigma=self.enforce_learning_sigma
             )

    def zero_grad(self,):
        """Zero the gradients of all optimized parameters."""
        self.model.zero_grad()


def mesu(model: Module, *, mu_prior: float, sigma_prior: float, N_mu: int, N_sigma: int, normalise_grad_sigma: int, normalise_grad_mu: int, c_sigma: float, c_mu: float, second_order: bool, clamp_sigma: list, clamp_mu: list, enforce_learning_sigma: int):

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
                square_root_sigma = 1e-12 + (normalise_grad_sigma == 0)*1 + (normalise_grad_sigma == 1)*(grad_sigma**2).mean()**0.5\
                    + (normalise_grad_sigma == 2)*(grad_mu**2).mean()**0.5 + (normalise_grad_sigma == 3)*((grad_sigma**2).mean()**0.5+(grad_mu**2).mean()**0.5)\
                    + (normalise_grad_sigma == 4)*(torch.clamp(grad_sigma, 0, 1e12)/sigma).mean() + \
                    (normalise_grad_sigma == 5) * \
                    (((torch.clamp(grad_sigma, 0, 1e12)/sigma)**2).mean()**0.5)
                square_root_mu = 1e-12 + (normalise_grad_mu == 0)*1 + (normalise_grad_mu == 1)*(grad_sigma**2).mean()**0.5 + (
                    normalise_grad_mu == 2)*(grad_mu**2).mean()**0.5 + (normalise_grad_mu == 3)*((grad_sigma**2).mean()**0.5+(grad_mu**2).mean()**0.5)
                grad_sigma = c_sigma*grad_sigma/square_root_sigma
                # print(grad_sigma.std().detach().cpu().numpy(), grad_sigma.shape,sigma.data.mean().detach().cpu().numpy())
                grad_mu = c_mu*grad_mu/square_root_mu
                denominator_sigma = 1 + second_order * \
                    (variance * grad_sigma ** 2)
                denominator_mu = 1 + second_order * (variance * grad_mu ** 2)
                sigma.data.add_(-0.5*(variance * grad_sigma + sigma.data * (
                    variance-sigma_prior ** 2) / (N_sigma * sigma_prior ** 2)) / denominator_sigma)
                mu.data.add_(-(variance * grad_mu + variance * (mu.data -
                             mu_prior) / (N_mu * sigma_prior ** 2)) / denominator_mu)
                if clamp_sigma[0] != 0:
                    sigma.data = torch.clamp(
                        sigma.data, clamp_sigma[0], clamp_sigma[1])
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(mu.data, clamp_mu[0], clamp_mu[1])

            if grad_sigma == None and grad_mu != None:
                denominator_mu = 1 + second_order*variance * grad_mu ** 2
                mu.data.add_(-(variance * grad_mu + variance * (mu.data -
                             mu_prior) / (N_mu * sigma_prior ** 2)) / denominator_mu)
                if clamp_mu[0] != 0:
                    mu.data = torch.clamp(mu.data, clamp_mu[0], clamp_mu[1])

            if grad_sigma == None and enforce_learning_sigma == True and grad_mu != None:

                grad_sigma = sigma * (grad_mu**2) / (grad_mu**2).mean()
                grad_sigma = c_sigma*grad_sigma
                denominator_sigma = 1 + second_order * \
                    (variance * grad_sigma ** 2)
                sigma.data.add_(-0.5*(variance * grad_sigma + sigma.data * (
                    variance-sigma_prior ** 2) / (N_sigma * sigma_prior ** 2)) / denominator_sigma)
                if clamp_sigma[0] != 0:
                    sigma.data = torch.clamp(
                        sigma.data, clamp_sigma[0], clamp_sigma[1])

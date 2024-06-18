import torch
from typing import Union


class BHUparallel(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr_mult: float = 1.0,
                 lr_max: float = 30,
                 likelihood_coeff: float = 1.0,
                 kl_coeff: float = 1.0,
                 normalize_gradients: bool = False,
                 eps: float = 1e-7,
                 clamp: float = 0.1,
                 mesuified: bool = False,
                 N: int = 1):
        """ Binary Bayesian optimizer for continual learning

        Args:
            lr_asymmetry: Inversely proportional to the asymmetry of the network (stronger means less assymetry)
            lr_max: Maximum learning rate that the system can have
            likelihood_coeff: Coefficient for the likelihood term
            kl_coeff: Coefficient for the KL divergence term
            normalize_gradients: Whether to normalize gradients
            eps: Small constant to avoid division by zero
            clamp: Clamping value for gradients
            mesuified: Whether to use the mesuified version of the optimizer
            N: Number of tasks maximally learned
        """

        defaults = dict(lr_mult=lr_mult,
                        lr_max=lr_max,
                        likelihood_coeff=likelihood_coeff,
                        kl_coeff=kl_coeff,
                        normalize_gradients=normalize_gradients,
                        eps=eps,
                        clamp=clamp,
                        mesuified=mesuified,
                        N=N)
        super(BHUparallel, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        lr_array = []
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1
                lr_mult = group['lr_mult']
                lr_max = group['lr_max']
                likelihood_coeff = group['likelihood_coeff']
                kl_coeff = group['kl_coeff']
                normalize_gradients = group['normalize_gradients']
                eps = group['eps']
                clamp = group['clamp']
                mesuified = group['mesuified']
                N = group['N']
                lambda_ = p.data
                # Normalize gradients if specified
                if normalize_gradients:
                    p.grad.data = p.grad.data / \
                        (torch.norm(p.grad.data, p=2) + eps)
                    p.grad.data = torch.clamp(p.grad.data, -clamp, clamp)
                # Update rule for lambda with Hessian correction
                kl = kl_coeff/torch.cosh(lambda_)**2
                likelihood = likelihood_coeff*2*p.grad.data*torch.tanh(lambda_)
                hessian = likelihood_coeff * 2 * \
                    torch.abs(p.grad.data) + 1/lr_max
                lr_asymmetry = 1/(kl + likelihood + hessian)

                # Update the weights
                if mesuified == False:
                    p.data = lambda_ - \
                        lr_mult * lr_asymmetry * p.grad.data
                else:
                    p.data = lambda_ - lr_mult * lr_asymmetry * (p.grad.data) \
                        - lr_mult * lr_asymmetry * lambda_ / \
                        (N*torch.cosh(lambda_)**2)
                lr_array.append(lr_asymmetry*lr_mult)
        self.state['lr'] = lr_array
        return loss

import torch
from typing import Union


class BHUparallel(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 metaplasticity: float = 1.0,
                 lr_max: float = 30,
                 ratio_coeff: float = 1.0,
                 mesuified: bool = False,
                 N: int = 1):
        """ Binary Bayesian optimizer for continual learning

        Args:
            metaplasticity: Metaplasticity parameter
            lr_max: Maximum learning rate imposed to the optimizer
            ratio_coeff: Ratio coefficient between KL and likelihood (KL/likelihood)
            normalize_gradients: Whether to normalize gradients
            eps: Small constant to avoid division by zero
            clamp: Clamping value for gradients
            mesuified: Whether to use the mesuified version of the optimizer
            N: Number of tasks maximally learned
        """

        defaults = dict(metaplasticity=metaplasticity,
                        lr_max=lr_max,
                        ratio_coeff=ratio_coeff,
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
                lr_max = group['lr_max']
                ratio_coeff = group['ratio_coeff']
                mesuified = group['mesuified']
                N = group['N']
                metaplasticity = group['metaplasticity']
                lambda_ = p.data
                # Update rule for lambda with Hessian correction
                kl = 1/torch.cosh(lambda_)**2
                likelihood = 2*p.grad.data*torch.tanh(lambda_)
                hessian = 2*p.grad.data.abs()
                lr_asymmetry = 1/(metaplasticity*(ratio_coeff*kl +
                                  likelihood + hessian) + 1/lr_max)
                # Update the weights
                if mesuified == True:
                    prior = torch.zeros_like(p.grad.data)
                    p.data = lambda_ - lr_asymmetry * \
                        (p.grad.data + (lambda_ - prior) /
                         (2*N*torch.cosh(lambda_)**2))
                else:
                    p.data = lambda_ - lr_asymmetry * p.grad.data
                lr_array.append(lr_asymmetry)
        self.state['lr'] = lr_array
        return loss

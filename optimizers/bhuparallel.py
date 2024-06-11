import torch
from typing import Union


class BHUparallel(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr_max: float = 30,
                 likelihood_coeff: float = 1.0,
                 kl_coeff: float = 1.0,
                 normalize_gradients: bool = False,):
        """ Binary Bayesian optimizer for continual learning

        Args:
            lr_asymmetry: Inversely proportional to the asymmetry of the network (stronger means less assymetry)
            lr_max: Maximum learning rate that the system can have
            likelihood_coeff: Coefficient for the likelihood term
            kl_coeff: Coefficient for the KL divergence term

        """

        defaults = dict(lr_max=lr_max,
                        likelihood_coeff=likelihood_coeff,
                        kl_coeff=kl_coeff,
                        normalize_gradients=normalize_gradients)
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
                likelihood_coeff = group['likelihood_coeff']
                kl_coeff = group['kl_coeff']
                normalize_gradients = group['normalize_gradients']
                lambda_ = p.data
                # Normalize gradients if specified
                if normalize_gradients:
                    p.grad.data = p.grad.data / \
                        (torch.norm(p.grad.data, p=2) + 1e-7)
                    # clip gradients
                    p.grad.data = torch.clamp(p.grad.data, -0.1, 0.1)
                # Update rule for lambda with Hessian correction
                asymmetry = 1/(kl_coeff*(1-torch.tanh(lambda_)**2) + likelihood_coeff*(2*p.grad.data*torch.tanh(lambda_) + 2 * torch.abs(p.grad.data)) + 1 / lr_max)
                lr_array.append(asymmetry)
                p.data = lambda_ - asymmetry * p.grad.data
        self.state['lr'] = lr_array
        return loss

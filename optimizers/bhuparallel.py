import torch
from typing import Union


class BHUparallel(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 likelihood_coeff: float = 1.0,
                 kl_coeff: float = 1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr,
                        likelihood_coeff=likelihood_coeff,
                        kl_coeff=kl_coeff)
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
                lr = group['lr']
                likelihood_coeff = group['likelihood_coeff']
                kl_coeff = group['kl_coeff']
                lambda_ = p.data
                gradient_estimate = p.grad.data
                # Update rule for lambda with Hessian correction
                hessian = 2*torch.abs(gradient_estimate) + 1/lr
                asymmetry = 1/(kl_coeff*(1-torch.tanh(lambda_)**2)+likelihood_coeff *
                               (2*gradient_estimate*torch.tanh(lambda_) + hessian))
                lr_array.append(asymmetry)
                p.data = lambda_ - asymmetry * gradient_estimate
        self.state['lr'] = lr_array
        return loss

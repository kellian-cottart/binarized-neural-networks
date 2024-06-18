import torch
from typing import Union


class BayesBiNNParallel(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr: Union[float, torch.Tensor] = 1e-4,
                 clamp_cosh: float = 30,
                 beta: float = 0,
                 scale: float = 1.0
                 ):
        """ Binary Bayesian optimizer

        Args:
            lr (float): Learning rate
            clamp_cosh (float): Maximum value for the squared hyperbolic cosine to avoid diverging
            beta (float): Momentum parameter
            scale (float): Scale parameter for the prior
        """

        defaults = dict(lr=lr,
                        clamp_cosh=clamp_cosh,
                        beta=beta,
                        scale=scale)
        super(BayesBiNNParallel, self).__init__(params, defaults)

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
                    state['momentum'] = torch.zeros_like(p.data)
                    state['prior_lambda'] = torch.zeros_like(p.data)
                state['step'] += 1
                # Get the parameters
                lr = group['lr']
                beta = group['beta']
                scale = group['scale']
                clamp_cosh = group['clamp_cosh']
                # Using the chain rule, we extend the gradient on lambda to retrieve the gradient on mu
                chain_rule = torch.clamp(torch.cosh(p.data)**2, max=clamp_cosh)
                state['momentum'] = beta * state['momentum'] + \
                    p.grad.data * chain_rule * (1 - beta)
                p.data = p.data - lr * \
                    state['momentum'] - lr * \
                    scale * (p.data - state['prior_lambda'])
                lr_array.append(chain_rule*lr)
        self.state['lr'] = lr_array
        return loss

    def update_prior_lambda(self):
        """ Update the prior lambda for continual learning
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['prior_lambda'] = p.data

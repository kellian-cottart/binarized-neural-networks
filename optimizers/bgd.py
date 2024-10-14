
from torch import Tensor, sqrt
from torch.optim.optimizer import Optimizer
from typing import List


class BGD(Optimizer):
    r"""
    Implements Bayesian Gradient Descent based on the work of Chen Zeno --> https://arxiv.org/pdf/1803.10123.

    Args:
        params: Model parameters. Parameters representing 'sigma' must be defined before 'mu'.
        lr: It should be 1. Can be greater to adjust the convergence rate. 
        clamp_grad: If >0, clamps the gradient of the loss over mu and sigma to this value, typically 0.1 or 1.

    Raises:
        ValueError: If input arguments are invalid.
    """

    def __init__(self, params, lr=1, clamp_grad=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, clamp_grad=clamp_grad)

        print("Careful! This optimizer takes only Meta Bayes parameters!")
        print("In your Module, sigma must be defined before mu")

        super().__init__(params, defaults)

        num_params = sum(len(group['params']) for group in self.param_groups)
        print(
            f'Optimizer initialized with {num_params} Gaussian variational parameters.')

    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            bgd(params_with_grad,
                d_p_list,
                lr=group['lr'],
                clamp_grad=group['clamp_grad'])


def bgd(params: List[Tensor], d_p_list: List[Tensor], lr: float, clamp_grad: float):
    if not params:
        raise ValueError('No gradients found in parameters!')
    if len(params) % 2 == 1:
        raise ValueError(
            'Parameters must include both Sigma and Mu in each group.')

    for sigma, mu, grad_sigma, grad_mu in zip(params[::2], params[1::2], d_p_list[::2], d_p_list[1::2]):
        if clamp_grad > 0:
            grad_sigma.data.clamp_(min=-clamp_grad/sigma, max=clamp_grad/sigma)
            grad_mu.data.clamp_(min=-clamp_grad/sigma, max=clamp_grad/sigma)
        variance = sigma.data ** 2
        sigma.data.add_(-0.5 * variance * grad_sigma - sigma *
                        (-1 + (1 + 0.25 * (variance * (grad_sigma ** 2))) ** 0.5))
        mu.data.add_(-lr * variance * grad_mu)

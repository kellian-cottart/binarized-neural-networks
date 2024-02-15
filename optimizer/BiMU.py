import torch
from torch.optim.optimizer import params_t
from typing import List, Union


class BinaryMetaplasticUncertainty(torch.optim.Optimizer):
    """ BinaryMetaplasticUncertainty (BiMU) Optimizer for PyTorch

    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate of the optimizer (default: 1e-3)
        gamma (float): coefficient of forgetting (default: 1e-6)
        temperature (float): temperature of the relaxation (changes the shape of the sigmoid)
        n_samples (int): number of MCMC n_samples to compute the gradient (default: 1)
        eps (float): term added to improve
            numerical stability
    """

    def __init__(self,
                 params: params_t,
                 lr: Union[float, torch.Tensor] = 1e-3,
                 scale: float = 1,
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= scale:
            raise ValueError(f"Invalid gamma: {scale}.")

        defaults = dict(lr=lr,
                        scale=scale)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

            BiMU(params,
                 grads,
                 lr=group['lr'],
                 scale=group['scale']
                 )
        return loss


def BiMU(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    lr: float,
    scale: float = 1,
):
    """ Perform a single optimization step"""
    for i, param in enumerate(params):
        # scale the gradient w.r.t the number of input samples
        grad = grads[i] * param.data.shape[0]
        condition = torch.where(torch.sign(param.data) != torch.sign(
            grad),
            torch.ones_like(param.data),  # STRENGTHENING
            scale)  # WEAKENING
        # sigma = 1 / torch.cosh(param.data)
        param.data = param.data - lr * grad * \
            condition * (1 - torch.tanh(param.data)**2)

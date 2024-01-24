import torch
from torch.optim.optimizer import params_t
from typing import List, Union


class BinaryMetaplasticUncertainty(torch.optim.Optimizer):
    """ BinaryMetaplasticUncertainty (BiMU) Optimizer for PyTorch

    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate of the optimizer (default: 1e-3)
        gamma (float): coefficient of forgetting (default: 1e-6)
        samples (int): number of MCMC samples to compute the gradient (default: 1)
        eps (float): term added to improve
            numerical stability
    """

    def __init__(self,
                 params: params_t,
                 lr: Union[float, torch.Tensor] = 1e-3,
                 gamma: float = 1e-6,
                 samples: int = 1,
                 eps: float = 1e-8,
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma: {gamma}.")
        if not 1 <= samples:
            raise ValueError(
                f"Invalid number of MCMC samples: {samples}")

        defaults = dict(lr=lr,
                        gamma=gamma,
                        samples=samples,
                        eps=eps)
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
                 samples=group['samples'],
                 lr=group['lr'],
                 gamma=group['gamma'],
                 eps=group['eps'])

        return loss


def BiMU(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    samples: int,
    lr: float,
    gamma: float,
    eps: float,
):
    """ Perform a single optimization step"""
    import time
    for i, param in enumerate(params):
        grad = grads[i]
        lambda_ = param.data
        mu = torch.tanh(param.data)
        sigma = 1 / torch.cosh(param.data)
        # 1. Sample from the uniform distribution U(0, 1) the logistic noise (G1 - G2)
        logistic_noise = torch.distributions.uniform.Uniform(0, 1).sample(
            (samples, *lambda_.shape)).to(lambda_.device)
        # 2. Compute delta = 1/2 * log(U/(1-U))
        delta = torch.log(logistic_noise / (1 - logistic_noise)) / 2
        # 3. Compute the relaxed weights
        relaxed_w = torch.tanh((lambda_ + delta))
        # 4. Compute the gradient of the binary weights w.r.t the mean
        scaling = ((1 - relaxed_w**2 + eps) /
                   ((1 - mu**2 + eps)))
        # 5. Compute the gradient estimate scaled by the number of samples
        gradient_estimate = torch.mean(scaling*grad, dim=0)*param.shape[0]

        # 6. Update the weights
        param.data = param.data - lr * sigma**2 * gradient_estimate

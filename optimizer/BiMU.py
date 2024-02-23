import torch
from torch.optim.optimizer import params_t
from typing import List, Union


class BinaryMetaplasticUncertainty(torch.optim.Optimizer):
    """ BinaryMetaplasticUncertainty (BiMU) Optimizer for PyTorch

    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate of the optimizer (default: 1e-3)
        gamma (float): coefficient of forgetting (default: 1e-6)
        noise (float): standard deviation of the normal distribution for the noise (default: 0)
        quantization (int): number of states between each integer (default: None)
        threshold (float): threshold for the values of the parameters (default: None)
    """

    def __init__(self,
                 params: params_t,
                 lr: Union[float, torch.Tensor] = 1e-3,
                 scale: float = 1,
                 noise: float = 0,
                 quantization: Union[int, None] = None,
                 threshold: Union[float, None] = None
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= scale:
            raise ValueError(f"Invalid gamma: {scale}.")

        defaults = dict(lr=lr,
                        scale=scale,
                        noise=noise,
                        quantization=quantization,
                        threshold=threshold
                        )
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
                 scale=group['scale'],
                 noise=group['noise'],
                 quantization=group['quantization'],
                 threshold=group['threshold']
                 )
        return loss


def BiMU(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    lr: float,
    scale: float = 1,
    noise: float = 0,
    quantization: Union[int, None] = None,
    threshold: Union[float, None] = None
):
    """ Perform a single optimization step"""
    for i, param in enumerate(params):
        # scale the gradient w.r.t the number of input samples
        grad = grads[i] * param.data.shape[0]

        sigma = torch.cosh(param.data)**2

        param.data -= lr * sigma*(1/(1+scale*torch.sign(param.data) *
                                     torch.sign(grad)))*grad

        if noise != 0:
            # create a normal distribution with mean lambda and std noise
            param.data += torch.distributions.normal.Normal(
                0, noise).sample(param.data.shape).to(param.data.device)
        if quantization is not None:
            # we want "quantization" states between each integer. For example, if quantization = 2, we want 0, 0.5, 1, 1.5, 2
            param.data = torch.round(
                param.data * quantization) / quantization
        if threshold is not None:
            # we want to clamp the values of lambda between -threshold and threshold
            param.data = torch.clamp(param.data, -threshold, threshold)

import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Magnetoionic(torch.optim.Optimizer):
    """ Magnetoionic Optimizer made for simulating magnetic fields in magnetoinic devices

    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        field (str, optional): type of magnetic field applied, either "strong", "weak" or "linear"
        eps (float, optional): term added to the denominator to improve numerical stability
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 field="weak",
                 eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr,
                        eps=eps)
        super(Magnetoionic, self).__init__(params, defaults)

        # Depending on the field applied, we have different polynomials
        if field == "strong":
            self.f_minus = lambda x: -100 + 200*torch.exp(-x/116)
            self.f_inverse_minus = lambda y: -116*torch.log((y+100)/200)

            self.f_plus = lambda x: 100 - 200*torch.exp(-x/80)
            self.f_inverse_plus = lambda y: -80*torch.log((100-y)/200)
        elif field == "weak":
            self.f_minus = lambda x: -100 + 200*torch.exp(-x/50)
            self.f_inverse_minus = lambda y: -50*torch.log((y+100)/200)

            self.f_plus = lambda x: 100 - 200*torch.exp(-x/131)
            self.f_inverse_plus = lambda y: -131*torch.log((100-y)/200)
        elif field == "linear":
            self.f_minus = lambda x: -x
            self.f_inverse_minus = lambda y: -y

            self.f_plus = lambda x: x
            self.f_inverse_plus = lambda y: y

        # import matplotlib.pyplot as plt
        # x = torch.linspace(-200, 200, 1000).cpu()
        # plt.plot(x, self.f_minus(x).cpu(), label="f_minus")
        # plt.plot(x, self.f_plus(x).cpu(), label="f_plus")
        # plt.legend()
        # plt.show()
        # exit()

    def __setstate__(self, state):
        super(Magnetoionic, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure(callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                lr = group['lr']
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1
                # Compute x = f^-1(w), the previous x of the old weights
                x = torch.where(grad >= 0, self.f_inverse_plus(
                    p.data), self.f_inverse_minus(p.data))
                # Compute the value of the polynom at the new position to get the new weights w_t = x_{t-1} + lr * grad
                w_t = torch.where(grad >= 0, self.f_plus(x - lr*torch.abs(grad)),
                                  self.f_minus(x - lr*torch.abs(grad)))
                # Update the parameters
                p.data = w_t
        return loss

import torch


class MetaplasticSGD(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 eps=1e-8,
                 metaplasticity=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr,
                        eps=eps,
                        metaplasticity=metaplasticity)
        super(MetaplasticSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MetaplasticSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.jit.export
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
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

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1
                metaplasticity = group['metaplasticity']
                lr = group['lr']
                condition = torch.where(p.data*grad > 0,
                                        1 -
                                        torch.tanh(metaplasticity *
                                                   torch.abs(p.data)),
                                        1)
                p.data -= lr*condition*grad
        return loss
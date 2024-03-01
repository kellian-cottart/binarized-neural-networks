import torch
from torch.optim.optimizer import params_t, _get_value, _dispatch_sqrt
from typing import List, Optional, Union, Tuple


class BinaryOptimizer(torch.optim.Optimizer):
    """ Binary Optimizer (BOP) for PyTorch

    Helwegen et al, Latent Weights Do Not Exist: Rethinking Binarized
    Neural Network Optimization

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        gamma (float): momentum factor (analogue to the learning rate)
        threshold (float): threshold for flipping the sign of the weights
        eps (float): term added to the denominator to improve
            numerical stability
        weight_decay (float): weight decay (L2 penalty)
        amsgrad (bool): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and
            Beyond`_
    """

    def __init__(self,
                 params: params_t,
                 gamma: float = 1e-2,
                 threshold: float = 1e-7,
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 maximize: bool = False,):
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= threshold:
            raise ValueError(f"Invalid threshold value: {threshold}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(gamma=gamma, threshold=threshold, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """ Set the optimizer state 

        Args:
            state (dict): State dictionary
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        """ Initialize the group before the optimization step

        Args:
            group (dict): Parameter group
            params_with_grad (list): List of parameters with gradients
            grads (list): List of gradients
            exp_avgs (list): List of exponential averages
            exp_avg_sqs (list): List of exponential averages of squared gradients
            max_exp_avg_sqs (list): List of maximum exponential averages of squared gradients
            state_steps (list): List of state steps
        """
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # Set initial step.
                    state['step'] = torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

    def step(self, closure=None):
        """ Perform a single optimization step 
        Taken from _single_tensor_adam in PyTorch

        Args: 
            closure (function): Function to evaluate loss
        """
        self._cuda_graph_capture_health_check()

        ### INITIALIZE GRADIENTS ###
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ### INITIALIZE GROUPS ###
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            gamma = group['gamma']
            threshold = group['threshold']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            ### ADAM OPTIMIZATION STEP ###
            binary_optimizer(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                found_inf=getattr(self, "found_inf", None),
                amsgrad=group['amsgrad'],
                gamma=gamma,
                threshold=threshold,
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                grad_scale=getattr(self, "grad_scale", None),
            )
        return loss


def binary_optimizer(params: List[torch.Tensor],
                     grads: List[torch.Tensor],
                     exp_avgs: List[torch.Tensor],
                     exp_avg_sqs: List[torch.Tensor],
                     max_exp_avg_sqs: List[torch.Tensor],
                     state_steps: List[torch.Tensor],
                     grad_scale: Optional[torch.Tensor],
                     found_inf: Optional[torch.Tensor],
                     *,
                     amsgrad: bool,
                     gamma: float,
                     threshold: float,
                     weight_decay: float,
                     eps: float,
                     maximize: bool):
    """ Perform a single optimization step
    Taken from _single_tensor_adam() in PyTorch
    """
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        momentum = exp_avgs[i]
        step_t = state_steps[i]

        # Update step
        step_t += 1
        if weight_decay != 0:
            grad = grad.add(param.data, alpha=weight_decay)

        # Decay momentum running average coefficient
        momentum.mul_(1-gamma).add_(grad.data, alpha=gamma)

        # Update the weights by flipping the sign of the weights if the threshold is reached
        param.data = -torch.sign(param.data *
                                 momentum - threshold) * param.data

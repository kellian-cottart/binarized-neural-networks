import torch
from torch.optim.optimizer import params_t, _get_value, _dispatch_sqrt
from typing import List, Optional, Union, Tuple


class SurrogateAdam(torch.optim.Optimizer):
    """ Surrogate Adam optimizer w/ hardtanh surrogate gradient
    Adding Metaplasticity to BNNs
    Directly taken from the definition of Adam in PyTorch

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        metaplasticity (float): metaplasticity value
        betas (Tuple[float, float]): coefficients used for computing
            running averages of gradient and its square
        eps (float): term added to the denominator to improve
            numerical stability
        weight_decay (float): weight decay (L2 penalty)
        amsgrad (bool): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and
            Beyond`_
        foreach (bool): whether to use a separate optimizer for each parameter
        maximize (bool): whether to maximize or minimize the loss function
        capturable (bool): whether the optimizer can be captured in a graph
        differentiable (bool): whether the optimizer is differentiable
    """

    def __init__(self,
                 params: params_t,
                 lr: Union[float, torch.Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 metaplasticity: float = 0.1,
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= metaplasticity:
            raise ValueError(f"Invalid metaplasticity value: {metaplasticity}")
        if isinstance(lr, torch.Tensor) and foreach and not capturable:
            raise ValueError(
                "lr as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, metaplasticity=metaplasticity, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """ Set the optimizer state 

        Args:
            state (dict): State dictionary
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
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
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group['capturable']
                        else torch.tensor(0.)
                    )
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
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError(
                        '`requires_grad` is not supported for `step` in differentiable mode')

                # Foreach without capturable does not support a tensor lr
                if group['foreach'] and torch.is_tensor(group['lr']) and not group['capturable']:
                    raise RuntimeError(
                        'lr as a Tensor is not supported for capturable=False and foreach=True')

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
            beta1, beta2 = group['betas']
            metaplasticity = group['metaplasticity']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            ### ADAM OPTIMIZATION STEP ###
            adam_metaplasticity(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                found_inf=getattr(self, "found_inf", None),
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                metaplasticity=metaplasticity,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                grad_scale=getattr(self, "grad_scale", None),
            )
        return loss


def adam_metaplasticity(params: List[torch.Tensor],
                        grads: List[torch.Tensor],
                        exp_avgs: List[torch.Tensor],
                        exp_avg_sqs: List[torch.Tensor],
                        max_exp_avg_sqs: List[torch.Tensor],
                        state_steps: List[torch.Tensor],
                        grad_scale: Optional[torch.Tensor],
                        found_inf: Optional[torch.Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        metaplasticity: float,
                        lr: Union[float, torch.Tensor],
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool):
    """ Perform a single optimization step
    Taken from _single_tensor_adam() in PyTorch, updated to support metaplasticity
    """

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                (param.is_cuda and step_t.is_cuda) or (
                    param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        # Update step
        step_t += 1
        if weight_decay != 0:
            grad = grad.add(param.data, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = _get_value(step_t)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        # Get the bias correction for the second moment
        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(
                max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt()).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt()).add_(eps)
        step_size = lr * bias_correction2_sqrt / bias_correction1
        # If the condition for the metaplastic update is met (i.e., the binary weights and the exponential average have the same sign), update the metaplasticity
        metaplastic_computation = torch.nn.functional.hardtanh(
            torch.mul(metaplasticity, param.data))
        metaplastic_condition = torch.mul(
            torch.sign(param).data, exp_avg) > 0.0
        if param.data.dim() == 1:
            # Update the bias
            param.data.addcdiv_(exp_avg, denom, value=-step_size)
        else:
            # Update the exponential average
            decayed_exp_avg = torch.mul(metaplastic_computation, exp_avg)
            # Update the weights with the metaplasticity
            param.data.addcdiv_(torch.where(metaplastic_condition,
                                            decayed_exp_avg, exp_avg), denom, value=-step_size)

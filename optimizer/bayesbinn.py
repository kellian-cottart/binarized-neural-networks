import torch
from torch.optim.optimizer import params_t, _get_value
from typing import List, Optional, Union
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BayesBiNN(torch.optim.Optimizer):
    """ BayesBiNN Optimizer for PyTorch

    Training Binary Neural Networks using the Bayesian Learning Rule, (Meng et al)

    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        beta (float): beta parameter to compute the running average of gradients
        temperature (float): temperature value of the Gumbel soft-max trick (Maddison et al., 2017)
        num_mcmc_samples (int): number of MCMC samples to compute the gradient (default: 1, if 0: computes the point estimate)
        prior_lambda (FloatTensor): lambda of the prior distribution (default: None)
    """

    def __init__(self,
                 params: params_t,
                 lr: Union[float, torch.Tensor] = 1e-3,
                 beta: float = 0.0,
                 temperature: float = 1.0,
                 num_mcmc_samples: int = 1,
                 prior_lambda: Optional[torch.Tensor] = None,
                 init_lambda: int = 10,
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")
        if prior_lambda is not None and not isinstance(prior_lambda, torch.Tensor):
            raise ValueError(
                f"Invalid prior lambda: {prior_lambda}, must be a tensor")
        if not 0 <= num_mcmc_samples:
            raise ValueError(
                f"Invalid number of MCMC samples: {num_mcmc_samples}")

        defaults = dict(lr=lr, beta=beta, temperature=temperature,
                        prior_lambda=prior_lambda, num_mcmc_samples=num_mcmc_samples, init_lambda=init_lambda)
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

                exp_avgs.append(state['exp_avg'])

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
            state_steps = []

            beta = group['beta']
            lr = group['lr']
            temperature = group['temperature']
            prior_lambda = group['prior_lambda']
            num_mcmc_samples = group['num_mcmc_samples']
            init_lambda = group['init_lambda']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                state_steps)

            ### ADAM OPTIMIZATION STEP ###
            bayes_optimization(
                params_with_grad,
                grads,
                exp_avgs,
                state_steps,
                beta=beta,
                lr=lr,
                temperature=temperature,
                num_mcmc_samples=num_mcmc_samples,
                prior_lambda=prior_lambda,
                init_lambda=init_lambda,
                closure=loss,
            )
        return loss


def bayes_optimization(params: List[torch.Tensor],
                       grads: List[torch.Tensor],
                       exp_avgs: List[torch.Tensor],
                       state_steps: List[torch.Tensor],
                       *,
                       beta: float,
                       lr: Union[float, torch.Tensor],
                       temperature: float,
                       num_mcmc_samples: int,
                       prior_lambda: torch.Tensor,
                       init_lambda: int,
                       eps: float = 1e-8,
                       closure: Optional[torch.Tensor] = None,
                       ):
    """ Perform a single optimization step

    The optimization step is based of the BayesBiNN algorithm (Meng et al)
    The general idea is to have an optimization step from a well-posed optimization problem using the Bayesian Learning Rule
    Using the Gumbel soft-max trick (Maddison et al., 2017), we can sample from the Bernoulli distribution to compute the gradient
    The implementation is based on Issam Laradji's implementation of the BayesBiNN algorithm

    Args:
        params (list): List of parameters
        grads (list): List of gradients
        exp_avgs (list): List of exponential averages
        state_steps (list): List of state steps
        beta (float): beta parameter to compute the running average of gradients
        lr (float): learning rate
        temperature (float): temperature value of the Gumbel soft-max trick (Maddison et al., 2017)
        prior_lambda (FloatTensor): lambda of the prior distribution (default: None)
        init_lambda (int): initial value of lambda
        num_mcmc_samples (int): number of MCMC samples to compute the gradient (default: 1, if 0: computes the point estimate)
    """
    # Sample epsilon according to uniform (0, 1)
    epsilon = torch.rand_like(params[0])
    # Set delta = 1/2 log (epsilon / (1 - epsilon))
    delta = 0.5 * torch.log(epsilon / (1 - epsilon))

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        state_step = state_steps[i]


class BiNNOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        model,
        train_set_size,
        N=5,
        lr=1e-9,
        temperature=1e-10,
        initialize_lambda=10,
        beta=0.99,
        *args,
        **kwargs,
    ):
        """
        For torch's Optimizer class
            Arguments:
            params (iterable): an iterable of :class:`torch.Tensor` s or
                :class:`dict` s. Specifies what Tensors should be optimized.
            defaults: (dict): a dict containing default values of optimization
                options (used when a parameter group doesn't specify them).
        """
        default_dict = dict(
            N=N,
            lr=lr,
            temperature=temperature,
            beta=beta,
            train_set_size=train_set_size,
        )

        super(BiNNOptimizer, self).__init__(model.parameters(), default_dict)

        # Natural parameter prior lambda = 0

        self.train_modules = []
        self.get_train_modules(model)

        self.param_groups[0]["lr"] = lr
        self.param_groups[0]["beta"] = beta
        p = parameters_to_vector(self.param_groups[0]["params"])

        # Initialization lamda  between -10 and 10
        # Convex combination
        theta1 = torch.randint_like(p, 2)
        self.state["lambda"] = (theta1 * initialize_lambda) - (
            (1 - theta1) * initialize_lambda
        )
        self.state["mu"] = torch.tanh(self.state["lambda"])
        self.state["momentum"] = torch.zeros_like(p)
        self.state["lambda_prior"] = torch.zeros_like(p)
        self.state["step"] = 0
        self.state["temperature"] = temperature

    def get_train_modules(self, model):
        """
        To get all the modules which have trainiable parameters.
        """
        if len(list(model.children())) == 0:
            if len(list(model.parameters())) != 0:
                self.train_modules.append(model)
        else:
            for sub_module in list(model.children()):
                self.get_train_modules(sub_module)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError(
                "Something is wrong in step function of optimizer class, Please Check!"
            )

        self.state["step"] += 1

        lr = self.param_groups[0]["lr"]
        parameters = self.param_groups[0]["params"]

        N = self.defaults["N"]
        beta = self.defaults["beta"]
        M = self.defaults["train_set_size"]

        mu = self.state["mu"]
        lamda = self.state["lambda"]

        temperature = self.defaults["temperature"]
        grad = torch.zeros_like(lamda)

        loss_list = []
        pred_list = []
        if N <= 0:
            relaxed_w = torch.tanh(self.state["lambda"])
            vector_to_parameters(relaxed_w, parameters)
            loss, pred = closure()
            pred_list.append(pred)
            loss_list.append(loss.detach())
            g_temp = torch.autograd.grad(loss, parameters)
            g = parameters_to_vector(g_temp).detach()
            grad = M * g
        else:
            for num in range(N):
                epsilon = torch.rand_like(mu)
                delta = torch.log(epsilon / (1 - epsilon)) / 2
                relaxed_w = torch.tanh(
                    (self.state["lambda"] + delta) / temperature)

                vector_to_parameters(relaxed_w, parameters)
                loss, pred = closure()
                pred_list.append(pred)
                loss_list.append(loss.detach())

                g = parameters_to_vector(
                    torch.autograd.grad(loss, parameters)).detach()
                s = (
                    (1 - relaxed_w * relaxed_w + 1e-10)
                    / temperature
                    / (1 - self.state["mu"] * self.state["mu"] + 1e-10)
                )
                grad.add_(s * g)

            grad.mul_(M / N)

        self.state["momentum"] = beta * self.state["momentum"] + (1 - beta) * (
            grad + self.state["lambda"]
        )  # P

        loss = torch.mean(torch.stack(loss_list))

        bias_correction1 = 1 - beta ** self.state["step"]

        self.state["lambda"] = (
            self.state["lambda"]
            - self.param_groups[0]["lr"] *
            self.state["momentum"] / bias_correction1
        )
        self.state["mu"] = torch.tanh(lamda)
        return loss, pred_list

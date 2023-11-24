import torch
from torch.optim.optimizer import params_t
from typing import Optional, Union
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BinarySynapticUncertainty(torch.optim.Optimizer):
    """ BayesBiNN Optimizer for PyTorch

    Training Binary Neural Networks using the Bayesian Learning Rule and aiming for metaplasticity


    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        beta (float): beta parameter to compute the running average of gradients
        temperature (float): temperature value of the Gumbel soft-max trick (Maddison et al., 2017)
        num_mcmc_samples (int): number of MCMC samples to compute the gradient (default: 1, if 0: computes the point estimate)
        prior_lambda (FloatTensor): lambda of the prior distribution (for continual learning, input the previously found distribution) (default: None)
        scale (float): scale of the prior distribution (default: 1)
    """

    def __init__(self,
                 params: params_t,
                 lr: Union[float, torch.Tensor] = 1e-4,
                 beta: float = 0.99,
                 temperature: float = 1e-8,
                 num_mcmc_samples: int = 1,
                 prior_lambda: Optional[torch.Tensor] = None,
                 init_lambda: int = 10,
                 scale: float = 1.0
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
                        prior_lambda=prior_lambda, num_mcmc_samples=num_mcmc_samples, scale=scale)
        super().__init__(params, defaults)

        ### LAMBDA INIT ###
        # We actually need the parameters of the first layer
        # to generate the bernouilli variable between 0 and 1
        param = parameters_to_vector(self.param_groups[0]['params'])
        bernouilli = torch.randint_like(param, 2)
        # Initialize lambda between -init_lambda and init_lambda
        self.state['lambda'] = (bernouilli * init_lambda) - \
            ((1 - bernouilli) * init_lambda)
        # Set all other parameters
        self.state['mu'] = torch.tanh(self.state['lambda'])
        self.state['step'] = 0
        self.state['momentum'] = torch.zeros_like(param)
        self.state['grad'] = torch.zeros_like(param)
        if prior_lambda is None:
            self.state['prior_lambda'] = torch.zeros_like(param)

    def update_prior_lambda(self):
        """ Update the prior lambda for continual learning
        """
        self.state['prior_lambda'] = self.state['lambda']

    def step(self, closure=None):
        """ Perform a single optimization step 
        Taken from _single_tensor_adam in PyTorch

        Args: 
            closure (function): Function to evaluate loss
        """
        self._cuda_graph_capture_health_check()

        # Necessity for the closure function to evaluate the loss
        if closure is None:
            raise RuntimeError(
                'BayesBiNN optimization step requires a closure function')
        loss = closure()

        self.state['step'] += 1

        ### INITIALIZE GROUPS ###
        # Groups are the iterable of parameters to optimize or dicts defining parameter groups
        for group in self.param_groups:

            # Parameters to optimize
            parameters = group['params']
            # Parameters of the optimizer
            eps = 1e-10
            beta = group['beta']
            lr = group['lr']
            temperature = group['temperature']
            num_mcmc_samples = group['num_mcmc_samples']
            scale = group['scale']

            # State of the optimizer
            step = self.state['step']
            grad = self.state['grad']
            lambda_ = self.state['lambda']
            mu = self.state['mu']
            momentum = self.state['momentum']
            prior_lambda = self.state['prior_lambda']

            ### OPTIMIZATION STEP ###
            if num_mcmc_samples <= 0:
                # Point estimate
                relaxed_w = torch.tanh(lambda_)
                vector_to_parameters(relaxed_w, parameters)
                g = parameters_to_vector(torch.autograd.grad(loss, parameters))
            else:
                # MCMC estimate
                for _ in range(num_mcmc_samples):
                    # Gumbel soft-max trick
                    epsilon = torch.rand_like(mu)
                    delta = torch.log(epsilon / (1 - epsilon)) / 2
                    relaxed_w = torch.tanh(
                        (lambda_ + delta) / temperature)
                    vector_to_parameters(relaxed_w, parameters)
                    # Compute the loss
                    loss = closure()
                    # Compute the gradient
                    g = parameters_to_vector(
                        torch.autograd.grad(loss, parameters))
                    s = ((1 - relaxed_w * relaxed_w + eps) / temperature /
                         (1 - mu * mu + eps))
                    grad.add_(s * g)
                grad.div_(num_mcmc_samples)

            # Update all parameters
            bias_correction = 1 - beta ** step
            step_size = lr / bias_correction

            momentum = momentum*beta + (1-beta)*grad
            lambda_ += step_size * \
                (scale*(prior_lambda - lambda_) - momentum)

            self.state['lambda'] = lambda_
            self.state['momentum'] = momentum
            self.state['mu'] = torch.tanh(lambda_)
        return loss

import torch
from torch.optim.optimizer import params_t
from typing import Optional, Union
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BinarySynapticUncertainty_OLD(torch.optim.Optimizer):
    """ BinarySynapticUncertainty Optimizer for PyTorch

    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        metaplasticity (float): learning rate of the metaplasticity (default: 1)
        lr (float): learning rate of the optimizer (default: 1e-3)
        temperature (float): temperature value of the Gumbel soft-max trick (Maddison et al., 2017)
        num_mcmc_samples (int): number of MCMC samples to compute the gradient (default: 1, if 0: computes the point estimate)
        init_lambda (int): initial value of lambda (default: 0)
        prior_lambda (torch.Tensor): prior value of lambda (default: None)
        gamma (float): coefficient of regularization (default: 0)
    """

    def __init__(self,
                 params: params_t,
                 metaplasticity: Union[float, torch.Tensor] = 1,
                 regularization_metaplasticity: Union[float, torch.Tensor] = 1,
                 lr: Union[float, torch.Tensor] = 1e-3,
                 temperature: float = 1e-8,
                 num_mcmc_samples: int = 1,
                 init_lambda: int = 0,
                 prior_lambda: Optional[torch.Tensor] = None,
                 gamma: float = 0.0
                 ):
        if not 0.0 <= metaplasticity:
            raise ValueError(f"Invalid learning rate: {metaplasticity}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")
        if not 0 <= num_mcmc_samples:
            raise ValueError(
                f"Invalid number of MCMC samples: {num_mcmc_samples}")
        if not 0.0 <= init_lambda:
            raise ValueError(f"Invalid initial lambda: {init_lambda}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma: {gamma}.")

        defaults = dict(metaplasticity=metaplasticity,
                        regularization_metaplasticity=regularization_metaplasticity,
                        lr=lr,
                        gamma=gamma,
                        temperature=temperature,
                        num_mcmc_samples=num_mcmc_samples)
        super().__init__(params, defaults)

        ### LAMBDA INIT ###
        # Know the size of the input
        param = parameters_to_vector(self.param_groups[0]['params'])
        # Initialize lambda as a gaussian around 0 with std init_lambda
        self.state['lambda'] = torch.normal(
            mean=0, std=init_lambda, size=param.shape
        ).to(param.device)
        # Set all other parameters
        self.state['mu'] = torch.tanh(self.state['lambda'])
        self.state['step'] = 0
        self.state['prior_lambda'] = []

        if prior_lambda is None:
            self.state['prior_lambda'].append(torch.zeros_like(param))
        else:
            self.state['prior_lambda'].append(prior_lambda)

    def update_prior_lambda(self):
        """ Update the prior lambda for continual learning
        """
        self.state['prior_lambda'] = self.state['lambda']

    def step(self, input_size=60_000, closure=None):
        """ Perform a single optimization step 
        Taken from _single_tensor_adam in PyTorch

        Args: 
            closure (function): Function to evaluate loss
            input_size (int): Size of the input data (default: 60_000)
        """
        self._cuda_graph_capture_health_check()

        # Necessity for the closure function to evaluate the loss
        if closure is None:
            raise RuntimeError(
                'BinarySynapticUncertainty optimization step requires a closure function')
        loss = closure()
        running_loss = []

        self.state['step'] += 1

        ### INITIALIZE GROUPS ###
        # Groups are the iterable of parameters to optimize or dicts defining parameter groups
        for i, group in enumerate(self.param_groups):
            if i != 0:
                break
            # Parameters to optimize
            parameters = group['params']
            # Parameters of the optimizer
            eps = 1e-10
            metaplasticity = group['metaplasticity']
            regularization_metaplasticity = group['regularization_metaplasticity']
            temperature = group['temperature']
            num_mcmc_samples = group['num_mcmc_samples']
            lr = group['lr']
            gamma = group['gamma']
            noise = group['noise']

            # State of the optimizer
            # lambda represents the intertia with each neuron
            lambda_ = self.state['lambda']
            mu = self.state['mu']
            self.state['prior_lambda'].append(lambda_)
            prior = self.state['prior_lambda'][0]

            gradient_estimate = torch.zeros_like(lambda_)

            if num_mcmc_samples <= 0:
                ### POINT ESTIMATE ###
                relaxed_w = torch.tanh(lambda_)
                vector_to_parameters(relaxed_w, parameters)
                loss = closure()
                running_loss.append(loss.item())
                g = parameters_to_vector(
                    torch.autograd.grad(loss, parameters)).detach()
                gradient_estimate = input_size * g
            else:
                ### MCMC SAMPLES ###
                for _ in range(num_mcmc_samples):
                    ### Gumbel soft-max trick ###
                    # Add eps to avoid log(0)
                    epsilon = torch.rand_like(lambda_) + eps
                    # Compute the exploration noise
                    delta = torch.log(epsilon / (1 - epsilon)) / 2
                    # Compute the relaxed weights
                    relaxed_w = torch.tanh(
                        (lambda_ + delta) / temperature)
                    # Update the parameters
                    vector_to_parameters(relaxed_w, parameters)
                    # Compute the loss
                    loss = closure()
                    running_loss.append(loss.item())
                    # Compute the gradient
                    g = parameters_to_vector(
                        torch.autograd.grad(loss, parameters)).detach()
                    s = ((1 - torch.pow(relaxed_w, 2) + eps) /
                         (temperature * (1 - torch.pow(mu, 2) + eps)))
                    gradient_estimate.add_(s * g)
                gradient_estimate.mul_(input_size).div_(
                    num_mcmc_samples if num_mcmc_samples > 0 else 1)

            ### PARAMETER UPDATE ###
            def metaplastic_func(m, x): return 1 / \
                torch.cosh(torch.mul(m, x)).pow(2)

            # Normal noise to add to the gradient
            uniform = torch.rand_like(gradient_estimate) * noise

            # Update lambda with metaplasticity
            lambda_ = lambda_ - (lr * metaplastic_func(metaplasticity, lambda_) *
                                 gradient_estimate + gamma *
                                 metaplastic_func(regularization_metaplasticity, prior -
                                 lambda_) * (lambda_)) * uniform

            # Use the prior lambda to coerce lambda
            self.state['lambda'] = lambda_
            self.state['mu'] = torch.tanh(lambda_)
            # remove the first element of the prior lambda
            self.state['prior_lambda'] = self.state['prior_lambda'][1:]
        return torch.mean(torch.tensor(running_loss))
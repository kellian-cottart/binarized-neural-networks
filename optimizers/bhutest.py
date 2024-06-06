import torch
from torch.optim.optimizer import params_t
from typing import Optional, Union
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BinaryHomosynapticUncertaintyTest(torch.optim.Optimizer):
    """ BinarySynapticUncertainty Optimizer for PyTorch

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate of the optimizer (default: 1e-3)
        beta (float): LTP/LTD ratio (default: 1) A higher value will increase the Long Term Depression (LTD) and decrease the Long Term Potentiation (LTP)
        gamma (float): prior learning rate term (default: 0)
        noise (float): noise added to the gradient (default: 0)
        temperature (float): temperature value of the Gumbel soft-max trick (Maddison et al., 2017)
        num_mcmc_samples (int): number of MCMC samples to compute the gradient (default: 1, if 0: computes the point estimate)
        init_lambda (int): initial value of lambda (default: 0)
        quantization (int): quantization of lambda (8 for 8 bits, 4 for 4 bits, etc.) (default: None)
        threshold (int): threshold to stop values of lambda (default: None)
        prior_lambda (torch.Tensor): prior value of lambda (default: None)
    """

    def __init__(self,
                 params: params_t,
                 lr: Union[float, torch.Tensor] = 1e-3,
                 likelihood_coeff: Optional[float] = 1,
                 kl_coeff: Optional[float] = 1,
                 beta: float = 0,
                 gamma: float = 0,
                 noise: float = 0,
                 temperature: float = 1,
                 num_mcmc_samples: int = 1,
                 init_law: str = "gaussian",
                 init_param: float = 0.1,
                 update: Optional[int] = 1,
                 quantization: Optional[int] = None,
                 threshold: Optional[int] = None,
                 prior_lambda: Optional[torch.Tensor] = None,
                 prior_attraction: Optional[float] = 0,
                 norm: Optional[bool] = False,
                 eps: float = 1e-1,
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= temperature:
            raise ValueError(f"Invalid temperature: {temperature}")
        if not 0 <= num_mcmc_samples:
            raise ValueError(
                f"Invalid number of MCMC samples: {num_mcmc_samples}")
        if not 0.0 <= noise:
            raise ValueError(f"Invalid noise: {noise}.")

        defaults = dict(lr=lr,
                        beta=beta,
                        gamma=gamma,
                        noise=noise,
                        temperature=temperature,
                        num_mcmc_samples=num_mcmc_samples,
                        threshold=threshold,
                        quantization=quantization,
                        update=update,
                        prior_attraction=prior_attraction,
                        norm=norm,
                        eps=eps,
                        likelihood_coeff=likelihood_coeff,
                        kl_coeff=kl_coeff
                        )
        super().__init__(params, defaults)

        ### LAMBDA INIT ###
        # Know the size of the input
        param = parameters_to_vector(
            [param for param in self.param_groups[0]['params'] if param.requires_grad])

        # gaussian distribution around 0
        if init_law == "gaussian":
            self.state['lambda'] = torch.distributions.normal.Normal(
                0, init_param).sample(param.shape).to(param.device) if init_param != 0 else torch.zeros_like(param)
        elif init_law == "uniform":
            self.state['lambda'] = torch.distributions.uniform.Uniform(
                -init_param/2, init_param/2).sample(param.shape).to(param.device) if init_param != 0 else torch.zeros_like(param)
        else:
            raise ValueError(
                f"Invalid initialization law: {init_law}. Choose between 'gaussian' and 'uniform'")
        # Set all other parameters
        self.state['step'] = 0
        self.state['prior_lambda'] = prior_lambda if prior_lambda is not None else torch.zeros_like(
            param)

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
                'BinarySynapticUncertainty optimization step requires a closure function')
        loss = closure()
        running_loss = []

        self.state['step'] += 1

        ### INITIALIZE GROUPS ###
        # Groups are the iterable of parameters to optimize or dicts defining parameter groups
        group = self.param_groups[0]

        # Parameters to optimize
        parameters = [param for param in group['params']
                      if param.requires_grad]
        # Parameters of the optimizer
        temperature = group['temperature']
        num_mcmc_samples = group['num_mcmc_samples']
        lr = group['lr']
        beta = group['beta']
        gamma = group['gamma']
        noise = group['noise']
        quantization = group['quantization']
        threshold = group['threshold']
        norm = group['norm']
        eps = group['eps']
        likelihood_coeff = group['likelihood_coeff']
        kl_coeff = group['kl_coeff']
        # State of the optimizer
        # lambda represents the intertia with each neuron
        lambda_ = self.state['lambda']
        prior = self.state['prior_lambda']

        def var(x):
            """ var of the tanh function"""
            return 1 - torch.tanh(x)**2

        if num_mcmc_samples == 0:
            ### POINT ESTIMATE ###
            relaxed_w = torch.tanh(lambda_)
            vector_to_parameters(relaxed_w, parameters)
            loss = closure()
            running_loss.append(loss.item())
            gradient_estimate = parameters_to_vector(
                torch.autograd.grad(loss, parameters)).detach()
        else:
            ### MCMC SAMPLES ###
            gradient_estimate = torch.zeros_like(lambda_)
            for _ in range(num_mcmc_samples):
                ### Gumbel soft-max trick ###
                # Add eps to avoid log(0)
                epsilon = torch.distributions.uniform.Uniform(
                    1e-10, 1).sample(lambda_.shape).to(lambda_.device)
                # Compute the logistic noise
                delta = torch.log(epsilon / (1 - epsilon)) / 2
                # Compute the relaxed weights
                z = (lambda_ + delta) / temperature
                relaxed_w = torch.tanh(z)
                # Update the parameters
                vector_to_parameters(relaxed_w, parameters)
                # Compute the loss
                loss = closure()
                running_loss.append(loss.item())
                # Compute the gradient at the relaxed weights
                g = parameters_to_vector(
                    torch.autograd.grad(loss, parameters)).detach()
                # Compute the scaling factor
                s = var(z)
                gradient_estimate.add_(s*g)
            gradient_estimate.div_(temperature * num_mcmc_samples)
        self.state["grad"] = gradient_estimate
        # if norm == True:
        #     self.state["grad"] = self.state["grad"] / \
        #         torch.norm(gradient_estimate, p=2)
        # METAPLASTICITY UPDATE
        hessian = 2*torch.abs(gradient_estimate) + 1/lr
        asymmetry = 1/(kl_coeff*(1-torch.tanh(lambda_)**2)+likelihood_coeff *
                       (2*gradient_estimate*torch.tanh(lambda_) + hessian))
        self.state["lr"] = asymmetry
        lambda_ = lambda_ - self.state["lr"]*self.state["grad"]
        # if noise != 0:
        #     # create a normal distribution with mean lambda and std noise
        #     lambda_ += torch.distributions.normal.Normal(
        #         0, noise).sample(lambda_.shape).to(lambda_.device)
        # if quantization is not None:
        #     # we want "quantization" states between each integer. For example, if quantization = 2, we want 0, 0.5, 1, 1.5, 2
        #     lambda_ = torch.round(
        #         lambda_ * quantization) / quantization
        # if threshold is not None:
        #     # we want to clamp the values of lambda between -threshold and threshold
        #     lambda_ = torch.clamp(lambda_, -threshold, threshold)
        self.state['lambda'] = lambda_
        return torch.mean(torch.tensor(running_loss))

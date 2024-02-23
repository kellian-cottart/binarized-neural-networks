import torch
from torch.optim.optimizer import params_t
from typing import Optional, Union
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class BinaryHomosynapticUncertaintyTest(torch.optim.Optimizer):
    """ BinarySynapticUncertainty Optimizer for PyTorch

    Args: 
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate of the optimizer (default: 1e-3)
        scale (float): LTP/LTD ratio (default: 1) A higher value will increase the Long Term Depression (LTD) and decrease the Long Term Potentiation (LTP)
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
                 scale: float = 1,
                 gamma: float = 0,
                 noise: float = 0,
                 temperature: float = 1,
                 num_mcmc_samples: int = 1,
                 init_lambda: int = 0,
                 update: Optional[int] = 1,
                 quantization: Optional[int] = None,
                 threshold: Optional[int] = None,
                 prior_lambda: Optional[torch.Tensor] = None,
                 ):
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
        if not 0.0 <= noise:
            raise ValueError(f"Invalid noise: {noise}.")

        defaults = dict(lr=lr,
                        scale=scale,
                        gamma=gamma,
                        noise=noise,
                        temperature=temperature,
                        num_mcmc_samples=num_mcmc_samples,
                        threshold=threshold,
                        quantization=quantization,
                        update=update,
                        )
        super().__init__(params, defaults)

        ### LAMBDA INIT ###
        # Know the size of the input
        param = parameters_to_vector(self.param_groups[0]['params'])

        # gaussian distribution around 0
        self.state['lambda'] = torch.distributions.normal.Normal(
            0, init_lambda).sample(param.shape).to(param.device) if init_lambda != 0 else torch.zeros_like(param)
        # Set all other parameters
        self.state['step'] = 0
        self.state['prior_lambda'] = prior_lambda if prior_lambda is not None else torch.zeros_like(
            param)

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
            temperature = group['temperature']
            num_mcmc_samples = group['num_mcmc_samples']
            lr = group['lr']
            scale = group['scale']
            gamma = group['gamma']
            noise = group['noise']
            quantization = group['quantization']
            threshold = group['threshold']
            update = group['update']

            # State of the optimizer
            # lambda represents the intertia with each neuron
            lambda_ = self.state['lambda']
            prior = self.state['prior_lambda']

            def meta(x, n):
                return 1/torch.cosh(x)**n

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
                gradient_estimate = torch.zeros_like(lambda_)
                for _ in range(num_mcmc_samples):
                    ### Gumbel soft-max trick ###
                    # Add eps to avoid log(0)
                    epsilon = torch.rand_like(lambda_) + eps
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
                    # Compute the gradient
                    g = parameters_to_vector(
                        torch.autograd.grad(loss, parameters)).detach()
                    s = meta(z, 2) / \
                        temperature*(meta(lambda_, 2))
                    gradient_estimate.add_(s*g)
                gradient_estimate.mul_(input_size).div_(
                    num_mcmc_samples if num_mcmc_samples > 0 else 1)

            if update == 1:
                # LAMBDA UPDATE WITH TANH + SIGN ON GRADIENT
                lambda_ -= lr * (1/(1+scale*torch.functional.F.tanh(lambda_) *
                                    torch.sign(gradient_estimate)))*gradient_estimate - lr * gamma * 1/(torch.cosh(1/lambda_)**2) * (prior - lambda_)
            elif update == 2:
                # LAMBDA UPDATE WITH HARDTANH + SIGN ON GRADIENT
                lambda_ -= lr * (1/(1+scale*torch.functional.F.hardtanh(lambda_, min_val=-1, max_val=1) *
                                    torch.sign(gradient_estimate)))*gradient_estimate - lr * gamma * 1/(torch.cosh(1/lambda_)**2) * (prior - lambda_)
            elif update == 3:
                # LAMBDA UPDATE WITH SIGN + SIGN ON GRADIENT
                lambda_ -= lr * (1/(1+scale*torch.sign(lambda_) *
                                    torch.sign(gradient_estimate)))*gradient_estimate - lr * gamma * 1/(torch.cosh(1/lambda_)**2) * (prior - lambda_)
            elif update == 4:
                # LAMBDA UPDATE WITH TANH + NOTHING ON GRADIENT
                lambda_ -= lr * (1/(1+scale*torch.tanh(lambda_) * gradient_estimate)) * \
                    gradient_estimate - lr * gamma * 1 / \
                    (torch.cosh(1/lambda_)**2) * (prior - lambda_)

            if noise != 0:
                # create a normal distribution with mean lambda and std noise
                lambda_ += torch.distributions.normal.Normal(
                    0, noise).sample(lambda_.shape).to(lambda_.device)
            if quantization is not None:
                # we want "quantization" states between each integer. For example, if quantization = 2, we want 0, 0.5, 1, 1.5, 2
                lambda_ = torch.round(
                    lambda_ * quantization) / quantization
            if threshold is not None:
                # we want to clamp the values of lambda between -threshold and threshold
                lambda_ = torch.clamp(lambda_, -threshold, threshold)

            self.state['lambda'] = lambda_
        return torch.mean(torch.tensor(running_loss))

    def visualize_grad(self, grad, threshold=1):
        """ Plot a graph with the distribution in lambda values with respect to certain thresholds

        Args:
            lambda_ (torch.Tensor): Lambda values
            path (str): Path to save the graph
            threshold (int): Threshold to plot the distribution
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        plt.figure()
        plt.grid()
        bins = 100
        hist = torch.histc(grad, bins=bins, min=-threshold,
                           max=threshold).detach().cpu()

        plt.bar(torch.linspace(-threshold, threshold, bins).detach().cpu(),
                hist * 100 / len(grad),
                width=0.01,
                zorder=2)
        plt.xlabel('Value of alpha*grad')
        plt.ylabel('% of alpha*grad')
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.gca().tick_params(which='both', width=1)
        plt.gca().tick_params(which='major', length=6)
        plt.ylim(0, 100)
        plt.show()

    def visualize_lambda(self, path, threshold=10):
        """ Plot a graph with the distribution in lambda values with respect to certain thresholds

        Args:
            lambda_ (torch.Tensor): Lambda values
            path (str): Path to save the graph
            threshold (int): Threshold to plot the distribution
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        import os
        from utils.visual import versionning
        plt.figure()
        plt.grid()
        bins = 100
        hist = torch.histc(self.state['lambda'], bins=bins, min=-threshold,
                           max=threshold).detach().cpu()

        plt.bar(torch.linspace(-threshold, threshold, bins).detach().cpu(),
                hist * 100 / len(self.state['lambda']),
                width=1.5,
                zorder=2)
        plt.xlabel('Value of $\lambda$ ')
        plt.ylabel('% of $\lambda$')
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.gca().tick_params(which='both', width=1)
        plt.gca().tick_params(which='major', length=6)
        plt.ylim(0, 100)

        textsize = 6
        transform = plt.gca().transAxes

        plt.text(0.5, 0.95, f"$\lambda$  Lambda values above {threshold}: {(self.state['lambda'] > threshold).sum() * 100 / len(self.state['lambda']):.2f}%",
                 fontsize=textsize, ha='center', va='center', transform=transform)
        plt.text(0.5, 0.9, f"$\lambda$ values above 2: {((self.state['lambda'] > 2) & (self.state['lambda'] < threshold)).sum() * 100 / len(self.state['lambda']):.2f}%",
                 fontsize=textsize, ha='center', va='center', transform=transform)
        plt.text(0.5, 0.85, f"$\lambda$  values below -2: {((self.state['lambda'] < -2) & (self.state['lambda'] > -threshold)).sum() * 100 / len(self.state['lambda']):.2f}%",
                 fontsize=textsize, ha='center', va='center', transform=transform)
        plt.text(0.5, 0.8, f"$\lambda$ values below -{threshold}: {(self.state['lambda'] < -threshold).sum() * 100 / len(self.state['lambda']):.2f}%",
                 fontsize=textsize, ha='center', va='center', transform=transform)
        plt.text(0.5, 0.75, f"$\lambda$ values between -2 and 2: {((self.state['lambda'] < 2) & (self.state['lambda'] > -2)).sum() * 100 / len(self.state['lambda']):.2f}%",
                 fontsize=textsize, ha='center', va='center', transform=transform)

        os.makedirs(path, exist_ok=True)
        plt.savefig(versionning(path, "lambda-visualization",
                    ".pdf"), bbox_inches='tight')
        plt.close()

import torch


class WeightSampler:
    """ Sampler for the weights of a Bayesian Linear Layer
    Samples weights for variational inference as in BBB paper

    Args:
        mean (torch.Tensor): Mean of the distribution
        std (torch.Tensor): Standard deviation of the distribution
    """

    def __init__(self, mean, std):
        # Parameters of the normal distribution
        self.mean = mean
        self.std = std
        # Normal distribution with mean 0 and std 1
        self.distribution = torch.distributions.Normal(0, 1)

    def sample(self, num_samples=1, log=False):
        """ Sample from the distribution according to the reparametrization trick

        Args:
            num_samples (int): Number of samples to generate (default: 1)
            log (bool): If True, returns mean + log(1+exp(std)) * epsilon instead of mean + std * epsilon

        Returns:
            torch.Tensor: Sampled weights
        """
        if num_samples < 0:
            raise ValueError('Number of samples must be positive')
        elif num_samples == 0:
            return self.mean

        # Sample from the normal distribution with a certain number of samples
        epsilon = self.distribution.sample(
            (num_samples, *self.mean.shape)).squeeze()
        # Reparametrization trick
        if log:
            return self.mean + torch.log(1 + torch.exp(self.std)) * epsilon
        return self.mean + self.std * epsilon


class BayesianLinear(torch.nn.Module):
    """ Bayesian Linear Layer

    Weights become a distribution with a mean and a standard deviation

    Args: 
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): If True, adds a learnable bias to the output
        device (str): Device to use for the computation
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, priors=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BayesianLinear, self).__init__()
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
            }
        elif not isinstance(priors, dict) or 'prior_mu' not in priors or 'prior_sigma' not in priors:
            raise TypeError(
                'Priors must be a dictionary containing prior_mu and prior_sigma')

        ### WEIGHTS ###
        self.weight_mean = torch.nn.Parameter(
            torch.Tensor(out_features, in_features), **factory_kwargs)
        self.weight_std = torch.nn.Parameter(
            torch.Tensor(out_features, in_features), **factory_kwargs)
        if bias:
            self.bias_mean = torch.nn.Parameter(
                torch.Tensor(out_features), **factory_kwargs)
            self.bias_std = torch.nn.Parameter(
                torch.Tensor(out_features), **factory_kwargs)
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_std', None)

        ### WEIGHTS SAMPLERS ###
        self.weight_sampler = WeightSampler(
            self.weight_mean, self.weight_std)
        if bias:
            self.bias_sampler = WeightSampler(self.bias_mean, self.bias_std)

        ### INITIALIZATION ###
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases
        """
        ### WEIGHTS ###
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        torch.nn.init.kaiming_uniform_(self.weight_mean, a=5**0.5)
        torch.nn.init.constant_(self.weight_std, 0.1)
        ### BIASES ###
        if self.bias_mean is not None:
            torch.nn.init.kaiming_uniform_(self.bias_mean, a=5**0.5)
            torch.nn.init.constant_(self.bias_std, 0.1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            input (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        ### SAMPLE WEIGHTS ###
        weight = self.weight_sampler.sample()
        if self.bias_mean is not None:
            bias = self.bias_sampler.sample()
        else:
            bias = None
        ### FORWARD PASS ###
        return torch.nn.functional.linear(input, weight, bias)

    def extra_repr(self):
        """String representation of the layer"""
        return "BayesianLinear(in_features={}, out_features={}, bias={})".format(
            self.in_features, self.out_features, self.bias_mean is not None
        )

import torch
from .activation import Sign


class BernouilliWeights:
    """ Bernouilli Weights

    Represents bayesian weights with a bernouilli distribution

    Args:
        lambda_ (torch.Tensor): Tensor of the same shape as the weights, represents the certainty of the weights of either being 1 or -1
    """

    def __init__(self, lambda_):
        """ Initialize the weights with a bernouilli distribution 

        Args:
            lambda_ (torch.Tensor): Tensor of the same shape as the weights, represents the certainty of the weights of either being 1 or -1. Will be sent as the weight parameter to the optimizer.
        """
        self.lambda_ = lambda_
        self.uniform = torch.distributions.uniform.Uniform(0, 1)

    def sample(self, samples=1):
        """ Sample from the exponential distribution using the Gumbel-softmax trick"""
        # 1. Sample from the uniform distribution U(0, 1) the logistic noise (G1 - G2)
        logistic_noise = self.uniform.sample(
            (samples, *self.lambda_.shape)).to(self.lambda_.device)
        # 2. Compute delta = 1/2 * log(U/(1-U))
        delta = (1/2 * torch.log(logistic_noise / (1 - logistic_noise))).to(
            self.lambda_.device)
        # 3. Compute the relaxed weights
        relaxed_w = torch.tanh((self.lambda_ + delta)).to(
            self.lambda_.device)
        # 4. Take the binary weights
        return Sign.apply(relaxed_w).to(
            self.lambda_.device)

    def sample_inference(self, samples=1):
        """ Sample from the bernouilli distribution using lambda"""
        # 1. Sample from the bernouilli distribution with p = sigmoid(2*lambda)
        bernouilli_noise = torch.distributions.bernoulli.Bernoulli(
            torch.sigmoid(2*self.lambda_)).sample((samples,)).to(self.lambda_.device)
        # 2. Scale the bernouilli noise to -1 and 1
        bernouilli_noise = 2 * bernouilli_noise - 1
        return Sign.apply(bernouilli_noise).to(
            self.lambda_.device)


class BayesianBiNNLinear(torch.nn.modules.Module):
    """ Binary Bayesian Linear Layer using the Gumbel-softmax trick

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        lambda_init (float): Initial value of the lambda parameter
        bias (bool): Whether to use a bias term
        device (torch.device): Device to use for the layer
        dtype (torch.dtype): Data type to use for the layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 lambda_init: float = 0.1,
                 bias: bool = False,
                 device: None = None,
                 dtype: None = None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BayesianBiNNLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameter lambda with a normal distribution of mean: 0 and std: lambda_init
        self.lambda_ = torch.nn.parameter.Parameter(torch.empty(
            (in_features, out_features), **factory_kwargs))
        if lambda_init != 0:
            self.lambda_.data = torch.distributions.normal.Normal(
                0, lambda_init).sample(self.lambda_.shape).to(self.lambda_.device)
        else:
            self.lambda_.data = torch.zeros_like(self.lambda_).to(
                self.lambda_.device)
        # Create the bernouilli weights from the lambda parameter
        self.weight = BernouilliWeights(self.lambda_)

    def forward(self,
                input: torch.Tensor,
                samples: int = 1):
        """ Forward pass of the layer

        Args:
            input (torch.Tensor): Input tensor
            samples (int): Number of samples to draw

        Returns:
            torch.Tensor: Output tensor
        """
        return torch.matmul(input, self.weight.sample(samples)).to(
            self.lambda_.device)

    def extra_repr(self):
        return 'in_features={}, out_features={}, lambda.shape={}'.format(
            self.in_features, self.out_features, self.lambda_.shape)

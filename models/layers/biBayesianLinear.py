import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class BinaryGumbelSoftmaxTrick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_, tau=1, samples=1):
        """Forward pass of the neural network"""
        # Compute epsilon from uniform U(0,1), but avoid 0
        epsilon = torch.distributions.Uniform(
            1e-10, 1).sample((samples, *lambda_.shape)).to(x.device)
        # Compute delta = 1/2 log(epsilon/(1-epsilon))
        delta = 0.5 * torch.log(epsilon/(1-epsilon))
        # Compute the new weights values
        weights = torch.tanh((lambda_ + delta)/tau)
        ctx.save_for_backward(x, weights)
        # Return the forward for this layer
        # soi: samples, out_features, in_features ; b, i: batch, in_features ; s, b, i: samples, batch, in_features
        return torch.einsum('soi, sbi -> sbo', weights, x)

    @staticmethod
    def backward(ctx, grad_output, tau=1):
        """ Backward propagation is using the Gumbel softmax trick"""
        # Retrieve the saved tensors
        x, weights = ctx.saved_tensors
        # Compute the mean normalization term for the Gumbel softmax
        normalization = (1-weights**2)/tau
        # Compute the gradient
        # soi: samples, out_features, in_features ; b, i: batch, in_features ; s, b, i: samples, batch, in_features
        grad_lambda = (torch.einsum('sbo, sbi -> soi', grad_output,
                                    x) * normalization).mean(0)
        grad_x = torch.einsum('soi, sbo -> sbi', weights, grad_output)
        return grad_x, grad_lambda, None, None


class BiBayesianLinear(torch.nn.Module):
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
                 tau: float = 1.0,
                 device: None = None,
                 dtype: None = None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BiBayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.lambda_ = nn.Parameter(torch.zeros(
            out_features, in_features, **factory_kwargs))

    def sample(self, x, n_samples=1):
        """ Sample the weights for the layer"""
        # Compute p for Bernoulli sampling
        p = torch.sigmoid(2*self.lambda_)
        # Sample the weights according to 2*Ber(p) - 1
        weights = 2*Bernoulli(p).sample((n_samples,)).to(x.device)-1
        # Return the forward for this layer
        # soi: samples, out_features, in_features ; b, i: batch, in_features ; s, b, i: samples, batch, in_features
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return torch.einsum('soi, sbi -> sbo', weights, x)

    def forward(self, x, n_samples=1):
        """ Forward pass of the neural network for the backward pass """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return BinaryGumbelSoftmaxTrick.apply(x, self.lambda_, self.tau, n_samples)

    def extra_repr(self):
        return 'in_features={}, out_features={}, lambda.shape={}'.format(
            self.in_features, self.out_features, self.lambda_.shape)

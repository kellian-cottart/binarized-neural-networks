from torch.nn import Parameter, Module
from torch import empty, empty_like


class GaussianParameter(Module):
    """Object used to perform the reparametrization tricks in gaussian sampling and reshape the tensor of samples in the right shape to prevents a for loop over the number of sample"""

    def __init__(self, out_features, in_features=None, kernel_size=None, **factory_kwargs):
        super(GaussianParameter, self).__init__()
        if in_features is None:
            self.sigma = Parameter(empty(
                (out_features,), **factory_kwargs))
        elif kernel_size is None:
            self.sigma = Parameter(
                empty((out_features, in_features), **factory_kwargs))
        else:
            self.sigma = Parameter(empty(
                (out_features, in_features, *kernel_size), **factory_kwargs))
        self.mu = Parameter(
            empty_like(self.sigma, **factory_kwargs))

    def sample(self, samples=1):
        """Sample from the Gaussian distribution using the reparameterization trick."""
        # Sample from the standard normal and adjust with sigma and mu
        if samples == 0:
            return self.mu.unsqueeze(0)
        sigma = self.sigma.repeat(samples, *([1] * (len(self.sigma.size()))))
        epsilon = empty_like(sigma).normal_()
        mu = self.mu.repeat(samples, *([1] * (len(self.mu.size()))))
        return mu + sigma * epsilon

    def extra_repr(self):
        return f"mu={self.mu.size()}, sigma={self.sigma.size()}"

class GaussianParameter:
    """Object used to perform the reparametrization tricks in gaussian sampling and reshape the tensor of samples in the right shape to prevents a for loop over the number of sample"""

    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu  # Mean of the distribution
        self.sigma = sigma  # Standard deviation of the distribution

    def sample(self, samples=1):
        """Sample from the Gaussian distribution using the reparameterization trick."""
        # Sample from the standard normal and adjust with sigma and mu
        if samples == 0:
            return self.mu.unsqueeze(0)
        buffer_epsilon = self.sigma.unsqueeze(0).repeat(
            samples, *([1]*len(self.sigma.shape)))
        epsilon = torch.empty_like(buffer_epsilon).normal_()
        mu = self.mu.unsqueeze(0).repeat(
            samples, *([1]*len(self.mu.shape)))
        return mu + buffer_epsilon * epsilon

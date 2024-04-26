
import jax.numpy as jnp
import jax
from flax import linen as nn
from dataloader import *


class SignActivation(nn.Module):
    """ Sign Activation Function

    Forward should be sign function
    Backward should be hardtanh function
    """

    def __call__(self, x):
        return jnp.sign(x)

    def grad(self, x):
        return jnp.clip(x, -1, 1)


class BinarizedDense(nn.Module):
    """ Binarized Dense Layer with Jax
    """

    def setup(self):
        self.w = self.param("w", nn.initializers.lecun_normal())
        self.b = self.param("b", nn.initializers.zeros)

    def __call__(self, x):
        x = jnp.dot(x, self.w)
        x = jnp.add(x, self.b)
        return x


class BNN(nn.Module):
    @nn.compact
    def __call__(self, layer_sizes, x):
        for i in range(len(layer_sizes) - 1):
            x = nn.Dense(layer_sizes[i + 1], name=f"dense_{i}")(x)
            x = nn.BatchNorm()(x)
        return x


if __name__ == "__main__":
    loader = GPULoading(padding=0,
                        device="cuda:0",
                        as_dataset=False)
    train, test, size, out = task_selection(
        loader,
        "MNIST",
        batch_size=32
    )
    layers = [784, 512, 512, out]
    key = jax.random.PRNGKey(0)

    print(SignActivation().grad(1))

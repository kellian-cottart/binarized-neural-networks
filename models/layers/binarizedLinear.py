import torch


class Sign(torch.autograd.Function):
    """ Sign Activation Function

    Allows for backpropagation of the sign function because it is not differentiable
    """

    @staticmethod
    def forward(ctx, tensor_input):
        ctx.save_for_backward(tensor_input)
        return tensor_input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_i = grad_output.clone()
        grad_i[i.abs() > 1.0] = 0
        return grad_i


class BinarizedLinear(torch.nn.Linear):
    """ Binarized Linear Layer

    Args:
        latent_weights (bool): Whether to use latent weights or not
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=False,
                 latent_weights=True,
                 device='cuda'
                 ):
        super(BinarizedLinear, self).__init__(
            in_features, out_features, bias=bias, device=device)
        self.latent_weights = latent_weights

    def forward(self, input):
        """Forward propagation of the binarized linear layer"""
        if not self.latent_weights:
            self.weight.data = self.weight.data.sign()
            if self.bias is not False and self.bias is not None:
                self.bias.data = self.bias.data.sign()
            return torch.nn.functional.linear(input, self.weight, self.bias)
        else:
            if self.bias is not None:
                return torch.nn.functional.linear(input, Sign.apply(self.weight), Sign.apply(self.bias))
            else:
                return torch.nn.functional.linear(input, Sign.apply(self.weight))

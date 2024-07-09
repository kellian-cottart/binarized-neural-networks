
import torch


class ElephantActivation(torch.nn.Module):
    """ Elephant Activation Layer

    Applies the elephant activation function to the input tensor.
    """

    def __init__(self, width=1, power=4):
        self.width = width
        self.power = power
        super(ElephantActivation, self).__init__()

    def forward(self, tensor_input):
        """ Forward pass of the Elephant function

        Args:
            tensor_input (torch.Tensor): Input tensor
            width (float): Width of the function
            power (float): Power of the function
        """
        return 1/(1+torch.absolute(tensor_input/self.width)**self.power)

    def extra_repr(self):
        return f"width={self.width}, power={self.power}"

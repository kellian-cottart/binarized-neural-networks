import torch
from tqdm import trange


class DNN(torch.nn.Module):
    """ Neural Network (DNN) 

    Neural Network with real-valued weights and activations.
    """

    def __init__(self, layers=[512], init='normal', std=0.01, device='cuda'):
        """ Initialize DNN

        Args: 
            layers (list): List of layer sizes (including input and output layers)
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
            device (str): Device to use for computation (e.g. 'cuda' or 'cpu')

        """
        super(DNN, self).__init__()
        self.n_layers = len(layers)-2
        self.layers = torch.nn.ModuleList()
        self.device = device

        ### LAYER INITIALIZATION ###
        for i in range(self.n_layers+1):
            # Linear layers with BatchNorm
            self.layers.append(torch.nn.Linear(
                layers[i], layers[i+1], bias=False, device=device))
            self.layers.append(torch.nn.BatchNorm1d(
                layers[i+1], affine=True, track_running_stats=True, device=device))

        ### WEIGHT INITIALIZATION ###
        for layer in self.layers[::2]:
            if init == 'gauss':
                torch.nn.init.normal_(
                    layer.weight, mean=0.0, std=std)
            elif init == 'uniform':
                torch.nn.init.uniform_(
                    layer.weight, a=-std/2, b=std/2)
            elif init == 'xavier':
                torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        """ Forward pass of DNN

        Args: 
            x (torch.Tensor): Input tensor

        Returns: 
            torch.Tensor: Output tensor

        """
        ### FORWARD PASS ###
        # For each pair of layers (linear + batchnorm)
        for linear, batchnorm in zip(self.layers[::2], self.layers[1::2]):
            x = linear(x)
            x = batchnorm(x)
            if batchnorm != self.layers[-1]:
                x = torch.nn.functional.relu(x)
        return x

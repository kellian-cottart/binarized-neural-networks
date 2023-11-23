import torch
from tqdm import trange


class DNN(torch.nn.Module):
    """ Neural Network Base Class
    """

    def __init__(self, layers=[512], init='normal', std=0.01, device='cuda', dropout=False, bias=False, *args, **kwargs):
        """ NN initialization

        Args: 
            layers (list): List of layer sizes (including input and output layers)
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
            device (str): Device to use for computation (e.g. 'cuda' or 'cpu')
            dropout (bool): Whether to use dropout
        """
        super(DNN, self).__init__()
        self.n_layers = len(layers)-2
        self.device = device
        self.layers = torch.nn.ModuleList()
        ### LAYER INITIALIZATION ###
        self._layer_init(layers, dropout, bias)
        ### WEIGHT INITIALIZATION ###
        self._weight_init(init, std)

    def _layer_init(self, layers, dropout=False, bias=False):
        """ Initialize layers of NN

        Args:
            dropout (bool): Whether to use dropout
            bias (bool): Whether to use bias
        """
        for i in range(self.n_layers+1):
            # Linear layers with BatchNorm
            if dropout and i != 0:
                self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.Linear(
                layers[i], layers[i+1], bias=bias, device=self.device))
            self.layers.append(torch.nn.BatchNorm1d(
                layers[i+1], affine=not bias, track_running_stats=True, device=self.device))

    def _weight_init(self, init='normal', std=0.01):
        """ Initialize weights of each layer

        Args:
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
        """
        for layer in self.layers:
            if hasattr(layer, 'weight'):
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
        for layer in self.layers:
            x = layer(x)
            if layer is not self.layers[-1] and isinstance(layer, torch.nn.BatchNorm1d):
                x = torch.nn.functional.relu(x)
        return torch.nn.functional.log_softmax(x, dim=1)

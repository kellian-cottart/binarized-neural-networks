import torch


class BinarizeLinear(torch.nn.Linear):
    """ Binarized Linear Layer

    Linear layer with binary weights and activations
    No bias term
    """

    def __init__(self):
        super(BinarizeLinear, self).__init__()
        if self.bias is not None:
            self.register_parameter('bias', None)


class BNN(torch.nn.Module):
    """ Binarized Neural Network (BNN) 

    Neural Network with binary weights and activations, using hidden weights called "degrees of certainty" (DOCs) to approximate real-valued weights.

    Axel Laborieux et al., Synaptic metaplasticity in binarized neural
networks
    """

    def __init__(self, layers=[512], init='normal', std=0.01, device='cuda'):
        """ Initialize BNN

        Args: 
            layers (list): List of layer sizes (including input and output layers)
            init (str): Initialization method for weights
            std (float): Standard deviation for initialization
            device (str): Device to use for computation (e.g. 'cuda' or 'cpu')

        """
        super(BNN, self).__init__()
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
        for i in range(self.n_layers+1):
            if init == 'gauss':
                torch.nn.init.normal_(
                    self.layers[i].weight, mean=0.0, std=std)
            elif init == 'uniform':
                torch.nn.init.uniform_(
                    self.layers[i].weight, a=-std/2, b=std/2)
            elif init == 'xavier':
                torch.nn.init.xavier_normal_(self.layers[i].weight)

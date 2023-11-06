
import torch
from .optimizer import SurrogateAdam


class BinarizedLinear(torch.nn.Linear):
    """ Binarized Linear Layer

    Linear layer with binary weights and activations
    Binarized Linear layer is an linear transformation: there is no bias term
    """

    def __init__(self, *args, **kwargs):
        super(BinarizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        """

        """
        # Put the binary weight in a buffer
        if not hasattr(self.weight, 'buffer'):
            self.weight.buffer = self.weight.data.clone()

        # Binarize the weights based on the sign of the hidden weights
        self.weight.data = torch.sign(self.weight.buffer)
        output = torch.nn.functional.linear(input, self.weight)

        # Remove bias
        if self.bias is not None:
            output -= self.bias.view(1, -1).expand_as(output)

        return output


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
            self.layers.append(BinarizedLinear(
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

    def forward(self, x):
        """Forward propagation of the binarized neural network"""
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                # Activation function is the sign function
                x = torch.sign(x).to(self.device)
        return x

    def train_network(self, train_data, n_epochs, learning_rate=0.01, metaplasticity=0, **kwargs):
        """Train the binarized neural network

        Args:
            train_data (torch.utils.data.DataLoader): Training dataset
            n_epochs (int): Number of epochs to train the network
            learning_rate (float): Learning rate for the optimizer

        Returns:
            list: List of accuracies for each epoch
        """
        ### OPTIMIZER ###
        optimizer = SurrogateAdam(
            self.parameters(), lr=learning_rate, metaplasticity=metaplasticity)
        loss_function = torch.nn.CrossEntropyLoss()
        accuracy_array = []
        ### TRAINING ###
        accuracy = []
        for epoch in range(n_epochs):
            # Set the network to training mode
            self.train()
            for x, y in train_data:
                # Forward pass
                x = x.view(x.shape[0], -1).to(self.device)
                y = y.to(self.device)
                y_hat = self.forward(x)

                # Loss function
                loss = loss_function(y_hat, y)

                # Backward pass
                # but we cannot backpropagate through the sign function using adam
                # we have to use a surrogate gradient using hardtanh
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ### EVALUATE ###
            accuracy = []
            if 'mnist_test' in kwargs:
                accuracy.append(self.test(kwargs['mnist_test']))
            if 'fashion_mnist_test' in kwargs:
                accuracy.append(self.test(kwargs['fashion_mnist_test']))
            accuracy_array.append(accuracy)

        return accuracy

    def test(self, data):
        """ Test DNN

        Args: 
            data (torch.utils.data.DataLoader): Testing data containing (data, labels) pairs 

        Returns: 
            float: Accuracy of DNN on data

        """
        ### ACCURACY ###
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data:
                # Flatten input
                x = x.view(x.shape[0], -1).to(self.device)
                y_pred = self.forward(x).to(self.device)
                # Retrieve the most likely class
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y.to(self.device)).sum().item()
        return correct/total

    def predict(self, x):
        """ Predict labels for data

        Args: 
            data (torch.Tensor): Data to predict labels for

        Returns: 
            torch.Tensor: Predicted labels

        """
        self.eval()
        ### PREDICT ###
        with torch.no_grad():
            x = x.view(1, -1).to(self.device)
            y_pred = self.forward(x).to(self.device)
            # Retrieve the most likely class
            _, predicted = torch.max(y_pred.data, 1)
        return predicted

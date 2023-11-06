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

    def save_state(self, path):
        """ Save state of DNN

        Args: 
            path (str): Path to save state

        """
        torch.save(self.state_dict(), path)

    def load_state(self, path):
        """ Load state of DNN

        Args: 
            path (str): Path to load state from

        """
        self.load_state_dict(torch.load(path))

    def train_network(self, train_data, n_epochs, learning_rate=0.01, weight_decay=0.01, **kwargs):
        """ Train DNN

        Args: 
            train_data (torch.utils.data.DataLoader): Training data containing (data, labels) pairs 
            test_data (torch.utils.data.DataLoader): Testing data containing (data, labels) pairs
            n_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for training
            *args: Additional arguments such as mnist_test, and fashion_mnist_test
        """
        ### OPTIMIZER ###
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_function = torch.nn.CrossEntropyLoss()
        accuracy_array = []
        ### TRAINING LOOP ###
        pbar = trange(n_epochs, desc='Initialization')
        for epoch in pbar:
            # Set model to training mode
            for i, (x, y) in enumerate(train_data):
                ### FORWARD PASS ###
                # Flatten input
                x = x.view(x.shape[0], -1).to(self.device)
                y = y.to(self.device)
                y_pred = self.forward(x)

                ### LOSS ###
                loss = loss_function(y_pred, y)
                ### BACKWARD PASS ###
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
            # Set postfix w/ all accuracies
            pbar.set_description(f"Epoch {epoch+1}/{n_epochs}")
            pbar.set_postfix(
                loss=loss.item(), mnist_test=accuracy[0], fashion_mnist_test=accuracy[1])
        return accuracy_array

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

from tqdm import trange
import torch


class Trainer:
    """Base class for all trainers."""

    def __init__(self, model, optimizer, optimizer_parameters, criterion, device):
        self.model = model
        self.optimizer = optimizer(model.parameters(), **optimizer_parameters)
        self.criterion = criterion
        self.device = device
        self.training_accuracy = []
        self.testing_accuracy = []

    def batch_step(self, inputs, targets):
        """Perform the training of a single sample of the batch
        """
        ### FORWARD PASS ###
        inputs = inputs.view(inputs.shape[0], -1).to(self.device)
        targets = targets.to(self.device)
        prediction = self.model.forward(inputs)

        ### LOSS ###
        self.loss = self.criterion(prediction, targets)

        ### BACKWARD PASS ###
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def epoch_step(self, train_loader, test_loader=None):
        """Perform the training of a single epoch
        """
        ### TRAIN WITH THE WHOLE BATCH ###
        for i, (inputs, targets) in enumerate(train_loader):
            self.batch_step(inputs, targets)

        ### EVALUATE ###
        if test_loader is not None:
            self.testing_accuracy.append(
                [self.test(data) for data in test_loader])

    def fit(self, train_loader, n_epochs, test_loader=None, verbose=True, **kwargs):
        """Train the model for n_epochs
        """
        pbar = trange(
            n_epochs, desc='Initialization') if verbose else range(n_epochs)
        for epoch in pbar:
            self.epoch_step(train_loader, test_loader)
            if verbose:
                pbar.set_description(f"Epoch {epoch+1}/{n_epochs}")
                pbar.set_postfix(current_loss=self.loss.item(
                ), mnist_test=self.testing_accuracy[-1][0], fashion_mnist_test=self.testing_accuracy[-1][1])

    def save(self, path):
        """Save the model
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load the model
        """
        self.model.load_state_dict(torch.load(path))

    @torch.no_grad()
    def predict(self, tensor):
        """ Predict labels for data

        Args: 
            data (torch.Tensor): Data to predict labels for

        Returns: 
            torch.Tensor: Predicted labels

        """
        tensor = tensor.view(1, -1).to(self.device)
        y_pred = self.model.forward(tensor).to(self.device)
        # Retrieve the most likely class
        _, predicted = torch.max(y_pred.data, 1)
        return predicted

    @torch.no_grad()
    def test(self, dataloader):
        """ Test DNN

        Args: 
            data (torch.utils.data.DataLoader): Testing data containing (data, labels) pairs 

        Returns: 
            float: Accuracy of DNN on data

        """
        ### ACCURACY COMPUTATION ###
        correct = 0
        total = 0
        for x, y in dataloader:
            x = x.view(x.shape[0], -1).to(self.device)
            y = y.to(self.device)
            y_pred = self.model.forward(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        return correct/total

from tqdm import trange
import torch
import wandb


class Trainer:
    """Base class for all trainers."""

    def __init__(self, model, optimizer, optimizer_parameters, criterion, device, logging=True, reduction="mean", *args, **kwargs):
        self.model = model
        self.optimizer = optimizer(
            self.model.parameters(), **optimizer_parameters)
        self.criterion = criterion
        self.reduction = reduction
        self.device = device
        self.training_accuracy = []
        self.testing_accuracy = []
        self.logging = logging
        # Scheduler addition
        if "scheduler" in kwargs:
            scheduler = kwargs["scheduler"]
            scheduler_parameters = kwargs["scheduler_parameters"]
            self.scheduler = scheduler(
                self.optimizer, **scheduler_parameters)

    def batch_step(self, inputs, targets):
        """Perform the training of a single batch
        """
        self.model.train()
        ### LOSS ###
        self.loss = self.criterion(
            self.model.forward(inputs.view(inputs.shape[0], -1)),
            targets,
            reduction=self.reduction)

        ### BACKWARD PASS ###
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def epoch_step(self, train_loader, test_loader=None):
        """Perform the training of a single epoch
        """
        ### SEND BATCH ###

        for inputs, targets in train_loader:
            self.batch_step(inputs, targets)

        ### SCHEDULER ###
        if "scheduler" in dir(self):
            self.scheduler.step()

        ### EVALUATE ###
        if test_loader is not None:
            # if we are testing with permuted MNIST, we need to assess all permutations
            if "test_permutations" in dir(self):
                self.testing_accuracy.append(self.test_continual(
                    test_loader[0]))
            # else, we just test the model on the testloader
            else:
                test = []
                for dataset in test_loader:
                    test.append(self.test(dataset))
                self.testing_accuracy.append(test)

        ### LOGGING ###
        if self.logging:
            self.log()

    def log(self):
        """Log the training and testing results for monitoring
        This function is called at the end of each epoch"""
        # loss
        wandb.log({"Loss": self.loss.item()})

        # training accuracy
        if len(self.training_accuracy) > 0:
            for task in range(len(self.testing_accuracy[-1])):
                wandb.log(
                    {f"Task {task+1} - Test accuracy": self.testing_accuracy[-1][task]})

        # learning rate
        if "lr" in self.optimizer.param_groups[0]:
            wandb.log({"Learning rate": self.optimizer.param_groups[0]['lr']})

    def fit(self, train_loader, n_epochs, test_loader=None, verbose=True, **kwargs):
        """Train the model for n_epochs
        """
        if verbose:
            pbar = trange(
                n_epochs, desc='Initialization')
        else:
            pbar = range(n_epochs)

        if "test_permutations" in kwargs:
            self.test_permutations = kwargs["test_permutations"]

        for epoch in pbar:
            self.epoch_step(train_loader, test_loader)

            ### PROGRESS BAR ###
            if verbose:
                pbar.set_description(f"Epoch {epoch+1}/{n_epochs}")
                # creation of a dictionnary with the name of the test set and the accuracy
                kwargs = {}
                if len(self.testing_accuracy) > 0:
                    kwargs = {
                        f"task {i+1}": f"{accuracy:.2%}" for i, accuracy in enumerate(self.testing_accuracy[-1])
                    }
                    # if number of task cannot fit in one line, print it in a new line
                if len(kwargs) > 4:
                    pbar.set_postfix(current_loss=self.loss.item(
                    ), lr=self.optimizer.param_groups[0]['lr'] if "lr" in self.optimizer.param_groups[0] else None)
                    # Do a pretty print of our results
                    pbar.write("=================")
                    pbar.write("Testing accuracy: ")
                    for key, value in kwargs.items():
                        pbar.write(f"\t{key}: {value}")
                else:
                    pbar.set_postfix(current_loss=self.loss.item(
                    ), **kwargs, lr=self.optimizer.param_groups[0]['lr'] if "lr" in self.optimizer.param_groups[0] else None)

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
        """ Predict labels for a single sample

        Args: 
            data (torch.Tensor): Data to predict labels for

        Returns: 
            torch.Tensor: Predicted labels
            torch.Tensor: Probability of predicted labels

        """
        tensor = tensor.view(1, -1).to(self.device)
        y_pred = self.model.forward(tensor).to(self.device)
        # Retrieve the most likely class from the softmax output
        _, predicted = torch.max(y_pred.data, 1)
        # Retrieve the probability of the most likely class
        probability = torch.nn.functional.softmax(
            y_pred.data, dim=1)[0][predicted]
        return predicted, probability

    @torch.no_grad()
    def test(self, dataloader):
        """ Predict labels for a full dataset and retrieve accuracy

        Args: 
            data (torch.utils.data.DataLoader): Testing data containing (data, labels) pairs 

        Returns: 
            float: Accuracy of DNN on data

        """
        ### ACCURACY COMPUTATION ###
        self.model.eval()
        # The test can be computed faster if the dataset is already in the right format
        if dataloader.dataset.data.dtype == torch.float32:
            x = dataloader.dataset.data.view(
                dataloader.dataset.data.shape[0], -1).to(self.device)
            predictions = self.model.forward(x).to(self.device)
            _, predicted = torch.max(predictions, dim=1)
        else:
            # For each batch of test (we only have one batch)
            for data, labels in dataloader:
                # Change according to the permutation
                x = data.view(data.shape[0], -1).squeeze().to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                predictions = self.model.forward(x).to(self.device)
                # Retrieve the most likely class from the softmax output
                _, predicted = torch.max(predictions, dim=1)
                # Retrieve the probability of the most likely class
        return torch.sum(predicted == labels) / len(dataloader.dataset.data)

    @torch.no_grad()
    def test_continual(self, dataloader):
        """ Test DNN with Permuted MNIST

        Args: 
            dataloader (torch.utils.data.DataLoader): Testing data containing (data, labels) pairs 
        """
        self.model.eval()
        accuracies = []
        labels = dataloader.dataset.targets.to(self.device)
        for permutation in self.test_permutations:
            if dataloader.dataset.data.dtype == torch.float32:
                x = dataloader.dataset.data.view(
                    dataloader.dataset.data.shape[0], -1)[:, permutation].to(self.device)
                predictions = self.model.forward(x).to(self.device)
                _, predicted = torch.max(predictions, dim=1)
            else:
                # For each batch of test (we only have one batch)
                for data, labels in dataloader:
                    # Change according to the permutation
                    x = data.view(data.shape[0], -1)[:,
                                                     permutation].squeeze().to(self.device)
                    labels = labels.to(self.device)
                    # Forward pass
                    predictions = self.model.forward(x).to(self.device)
                    # Retrieve the most likely class from the softmax output
                    _, predicted = torch.max(predictions, dim=1)
                    # Retrieve the probability of the most likely class
            accuracies.append(torch.sum(predicted == labels) /
                              len(dataloader.dataset.data))
        return accuracies

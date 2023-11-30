import torch
from tqdm import trange


class GPUTrainer:
    """Trainer that does not require the usage of DataLoaders

    Args: 
        model (torch.nn.Module): Model to train
        optimizer (torch.optim): Optimizer to use
        optimizer_parameters (dict): Parameters of the optimizer
        criterion (torch.nn): Loss function
        device (torch.device): Device to use for the training
        logarithmic (bool, optional): If True, the model outputs log probabilities. Defaults to True.
        reduction (str, optional): Reduction to use for the loss. Defaults to "mean".
        kwargs: Additional arguments
            scheduler (torch.optim.lr_scheduler, optional): Scheduler to use. Defaults to None.
            scheduler_parameters (dict, optional): Parameters of the scheduler. Defaults to None.
    """

    def __init__(self, model, optimizer, optimizer_parameters, criterion, device, logarithmic=True, reduction="mean", *args, **kwargs):
        self.model = model
        self.optimizer = optimizer(
            self.model.parameters(), **optimizer_parameters)
        self.criterion = criterion
        self.reduction = reduction
        self.device = device
        self.training_accuracy = []
        self.testing_accuracy = []
        self.logarithmic = logarithmic
        # Scheduler addition
        if "scheduler" in kwargs:
            scheduler = kwargs["scheduler"]
            scheduler_parameters = kwargs["scheduler_parameters"]
            self.scheduler = scheduler(
                self.optimizer, **scheduler_parameters)

    def batch_step(self, inputs, targets):
        """Perform the training of a single batch

        Args: 
            inputs (torch.Tensor): Input data
            targets (torch.Tensor): Labels
        """
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        ### LOSS ###
        self.loss = self.criterion(
            self.model.forward(inputs).to(self.device),
            targets.to(self.device),
            reduction=self.reduction)

        ### BACKWARD PASS ###
        self.loss.backward()
        self.optimizer.step()

    def epoch_step(self, train_dataset, test_loader=None):
        """Perform the training of a single epoch

        Args: 
            train_dataset (torch.Tensor): Training data
            test_loader (torch.Tensor, optional): Testing data. Defaults to None.
        """
        ### SEND BATCH ###

        for inputs, targets in train_dataset:
            if len(inputs.shape) == 4:
                # remove all dimensions of size 1
                inputs = inputs.squeeze()
            self.batch_step(inputs.to(self.device), targets.to(self.device))

        ### SCHEDULER ###
        if "scheduler" in dir(self):
            self.scheduler.step()

        ### EVALUATE ###
        if test_loader is not None:
            # if we are testing with permuted MNIST, we need to assess all permutations
            if "test_permutations" in dir(self):
                for testset in test_loader:
                    for inputs, targets in testset:
                        self.testing_accuracy.append(
                            self.test_continual(inputs.to(self.device),
                                                targets.to(self.device)))
            # else, we just test the model on the testloader
            else:
                test = []
                for testset in test_loader:
                    for inputs, targets in testset:
                        if len(inputs.shape) == 4:
                            # remove all dimensions of size 1
                            inputs = inputs.squeeze()
                        test.append(self.test(inputs.to(self.device),
                                              targets.to(self.device)))
                self.testing_accuracy.append(test)

    def test(self, inputs, labels):
        """ Predict labels for a full dataset and retrieve accuracy

        Args: 
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels of the data

        Returns: 
            float: Accuracy of the model on the dataset

        """
        ### ACCURACY COMPUTATION ###
        # The test can be computed faster if the dataset is already in the right format
        with torch.no_grad():
            # if self.forward can take log in its parameters
            if "log" in self.model.forward.__code__.co_varnames:
                predictions = self.model.forward(
                    inputs, log=self.logarithmic).to(self.device)
            else:
                predictions = self.model.forward(
                    inputs).to(self.device)
            idx_pred = torch.argmax(predictions, dim=1)
        return len(torch.where(idx_pred == labels)[0]) / len(labels)

    def test_continual(self, inputs, labels):
        """Test the model on the test set of the PermutedMNIST task

        Args:
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels of the data

        Returns:
            list: List of accuracies for each permutation
        """
        accuracies = []
        for permutation in self.test_permutations:
            x = inputs[:, permutation].to(self.device)
            accuracies.append(self.test(x, labels))
        return accuracies

    def save(self, path):
        """Save the model
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load the model
        """
        self.model.load_state_dict(torch.load(path))

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

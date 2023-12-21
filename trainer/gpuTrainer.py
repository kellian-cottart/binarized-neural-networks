import torch
import tqdm


class GPUTrainer:
    """Trainer that does not require the usage of DataLoaders

    Args: 
        model (torch.nn.Module): Model to train
        optimizer (torch.optim): Optimizer to use
        optimizer_parameters (dict): Parameters of the optimizer
        criterion (torch.nn): Loss function
        device (torch.device): Device to use for the training
        kwargs: Additional arguments
            scheduler (torch.optim.lr_scheduler, optional): Scheduler to use. Defaults to None.
            scheduler_parameters (dict, optional): Parameters of the scheduler. Defaults to None.
    """

    def __init__(self, model, optimizer, optimizer_parameters, criterion, device, *args, **kwargs):
        self.model = model
        self.optimizer = optimizer(
            self.model.parameters(),
            **optimizer_parameters
        )
        self.criterion = criterion
        self.device = device
        self.training_accuracy = []
        self.testing_accuracy = []
        self.mean_testing_accuracy = []
        # Scheduler addition
        if "scheduler" in kwargs:
            scheduler = kwargs["scheduler"]
            scheduler_parameters = kwargs["scheduler_parameters"]
            self.scheduler = scheduler(
                self.optimizer, **scheduler_parameters)

    def reset_optimizer(self, optimizer_parameters):
        """Reset the optimizer parameters such as momentum and learning rate

        Args:
            optimizer_parameters (dict): Parameters of the optimizer
        """
        self.optimizer = self.optimizer.__class__(
            self.model.parameters(), **optimizer_parameters)

    def batch_step(self, inputs, targets):
        """Perform the training of a single batch

        Args: 
            inputs (torch.Tensor): Input data
            targets (torch.Tensor): Labels
        """
        ### LOSS ###
        self.loss = self.criterion(
            self.model.forward(inputs).to(self.device),
            targets.to(self.device)
        )

        ### BACKWARD PASS ###
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def epoch_step(self, train_dataset):
        """Perform the training of a single epoch

        Args: 
            train_dataset (torch.Tensor): Training data
            test_loader (torch.Tensor, optional): Testing data. Defaults to None.
        """
        ### SEND BATCH ###
        self.model.train()
        for inputs, targets in train_dataset:
            if len(inputs.shape) == 4:
                # remove all dimensions of size 1
                inputs = inputs.squeeze()
            self.batch_step(inputs.to(self.device), targets.to(self.device))

    @torch.no_grad()
    def evaluate(self, test_loader):
        """ Evaluate the model on the test sets

        Args:
            test_loader (torch.utils.data.DataLoader): Testing data

        Returns:
            float: mean accuracy on the test sets
        """
        self.model.eval()
        with torch.no_grad():
            if test_loader is not None:
                test = []
                # Iterate over the Dataloaders
                for dataloader in test_loader:
                    batch = []
                    for inputs, targets in dataloader:
                        batch.append(
                            self.test(inputs.to(self.device), targets.to(self.device)))
                    test.append(torch.mean(torch.tensor(batch)))
                self.testing_accuracy.append(test)
                self.mean_testing_accuracy.append(
                    torch.mean(torch.tensor(test)))

    def fit(self, train_loader, n_epochs, test_loader=None, name_loader=None, permutations=None, **kwargs):
        """Train the model for n_epochs

        Args:
            train_loader (torch.utils.data.DataLoader): Training data
            n_epochs (int): Number of epochs
            test_loader (torch.utils.data.DataLoader, optional): Testing data. Defaults to None.
            name_loader (str, optional): Name of the test sets to print. Defaults to None.
            permutations (list, optional): Permutations to use for the permuted MNIST. Defaults to None.
        """
        pbar = tqdm.trange(n_epochs)
        for epoch in pbar:
            ### TRAINING ###
            self.epoch_step(train_loader)

            ### SCHEDULER ###
            if "scheduler" in dir(self):
                self.scheduler.step()

            ### TASK EVALUATION (+ PERMUTATION IF NEEDED) ###
            if permutations is not None:
                self.evaluate(self.yield_permutation(
                    test_loader[0], permutations))
            else:
                self.evaluate(test_loader)

            ### PROGRESS BAR ###
            self.pbar_update(pbar, epoch, n_epochs, name_loader)

    @torch.no_grad()
    def predict(self, inputs):
        """Predict the labels of the given inputs

        Args:
            inputs (torch.Tensor): Input data

        Returns:
            torch.Tensor: Predictions
        """
        self.model.eval()
        predictions = self.model.forward(
            inputs).to(self.device)
        return predictions

    @torch.no_grad()
    def test(self, inputs, labels):
        """ Predict labels for a full dataset and retrieve accuracy

        Args:
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels of the data

        Returns:
            float: Accuracy of the model on the dataset

        """
        ### ACCURACY COMPUTATION ###
        self.model.eval()
        predictions = self.predict(inputs)
        if "output_activation" in dir(self.model) and self.model.output_activation == "sigmoid":
            # apply exponential to get the probability
            predictions = torch.functional.F.sigmoid(predictions)
            predictions = torch.where(predictions >= 0.5, torch.ones_like(
                predictions), torch.zeros_like(predictions))
        else:
            predictions = torch.argmax(predictions, dim=1)
        return torch.mean((predictions == labels).float())

    def pbar_update(self, pbar, epoch, n_epochs, name_loader=None):
        """Update the progress bar with the current loss and accuracy"""
        pbar.set_description(f"Epoch {epoch+1}/{n_epochs}")
        # creation of a dictionnary with the name of the test set and the accuracy
        kwargs = {}
        if len(self.testing_accuracy) > 0:
            if name_loader is not None and len(self.testing_accuracy[-1]) != len(name_loader):
                raise ValueError(
                    "Not enough names for the test sets provided"
                )
            if name_loader is None:
                kwargs = {
                    f"task {i+1}": f"{accuracy:.2%}" for i, accuracy in enumerate(self.testing_accuracy[-1])
                }
            else:
                kwargs = {
                    name: f"{accuracy:.2%}" for name, accuracy in zip(name_loader, self.testing_accuracy[-1])
                }
            pbar.set_postfix(loss=self.loss.item())
            # Do a pretty print of our results
            pbar.write("=================")
            pbar.write("Testing accuracy: ")
            for key, value in kwargs.items():
                pbar.write(f"\t{key}: {value}")

    def save(self, path):
        """Save the model
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load the model
        """
        self.model.load_state_dict(torch.load(path))

    @staticmethod
    def yield_permutation(loader, permutations):
        """Yield the permuted inputs

        Args:
            loader (torch.DataLoader): Loader to use containing MNIST

        Yields:
            iterator: Iterator over the permuted loaders 
        """
        for permutation in permutations:
            loader.__unpermute__()
            yield loader.__permute__(permutation)

from trainer import Trainer
import torch


class GPUTrainer(Trainer):
    """Trainer that does not require the usage of DataLoaders

    Args: 
        batch_size (int): Batch size
        logarithmic (bool, optional): If True, the model outputs log probabilities. Defaults to True.
    """

    def __init__(self, batch_size, logarithmic=True, *args, **kwargs):
        self.batch_size = batch_size
        self.logarithmic = logarithmic
        super().__init__(*args, **kwargs)

    def epoch_step(self, train_dataset, test_loader=None):
        """Perform the training of a single epoch

        Args: 
            train_dataset (torch.Tensor): Training data
            test_loader (torch.Tensor, optional): Testing data. Defaults to None.
        """
        ### PERMUTE DATASET ###
        perm = torch.randperm(len(train_dataset))
        train_dataset.data = train_dataset.data[perm]
        train_dataset.targets = train_dataset.targets[perm]

        ### SEND BATCH ###
        n_batches = len(train_dataset) // self.batch_size
        for batch in range(n_batches):
            inputs = train_dataset[batch *
                                   self.batch_size:(batch+1)*self.batch_size][0]
            targets = train_dataset[batch *
                                    self.batch_size:(batch+1)*self.batch_size][1]
            self.batch_step(inputs, targets)

        ### SCHEDULER ###
        if "scheduler" in dir(self):
            self.scheduler.step()

        ### EVALUATE ###
        if test_loader is not None:
            # if we are testing with permuted MNIST, we need to assess all permutations
            if "test_permutations" in dir(self):
                self.testing_accuracy.append(self.test_continual(
                    test_loader[0].data, test_loader[0].targets))
            # else, we just test the model on the testloader
            else:
                test = []
                for dataset in test_loader:
                    inputs = dataset.data
                    labels = dataset.targets
                    test.append(self.test(inputs, labels))
                self.testing_accuracy.append(test)

        ### LOGGING ###
        if self.logging:
            self.log()

    def test(self, inputs, labels):
        """ Predict labels for a full dataset and retrieve accuracy

        Args: 
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels of the data
            log (bool, optional): If True, the model outputs log probabilities. Defaults to True.

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

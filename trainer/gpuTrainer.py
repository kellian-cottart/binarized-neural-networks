from trainer import Trainer
import torch


class GPUTrainer(Trainer):
    """Trainer that does not require the usage of DataLoaders"""

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)

    def epoch_step(self, train_dataset, test_loader=None):
        """Perform the training of a single epoch
        """

        ### SEND BATCH ###
        n_batches = len(train_dataset) // self.batch_size
        for batch in range(n_batches):
            inputs, targets = train_dataset[batch *
                                            self.batch_size:(batch+1)*self.batch_size]
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
                    test.append(self.test(dataset))
                self.testing_accuracy.append(test)

        ### LOGGING ###
        if self.logging:
            self.log()

    def test_continual(self, inputs, labels):
        """Test the model on the test set of the PermutedMNIST task

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader of the test set
        """
        self.model.eval()
        accuracies = []
        for permutation in self.test_permutations:
            x = inputs.view(inputs.shape[0], -1)[:,
                                                 permutation].to(self.device)
            predictions = self.model.forward(x).to(self.device)
            _, predicted = torch.max(predictions, dim=1)
            accuracies.append(torch.sum(predicted == labels) /
                              len(inputs))
        return accuracies

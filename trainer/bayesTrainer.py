from trainer.gpuTrainer import GPUTrainer
import torch


class BayesTrainer(GPUTrainer):
    """Extended Trainer class to cover the special case of BayesBiNN

    Necessity to have a different training function to implement mu and lambda properly 

    Args:
        Trainer (Trainer): Trainer class to extend
        *args: Variable length argument list (for the Trainer class)
        **kwargs: Arbitrary keyword arguments (most likely optimizer or scheduler parameters)
    """

    def __init__(self, *args, **kwargs):
        if "test_mcmc_samples" in kwargs["training_parameters"]:
            self.test_mcmc_samples = kwargs["training_parameters"]["test_mcmc_samples"]
        else:
            raise ValueError(
                "BayesTrainer needs test_mcmc_samples to be defined in training_parameters (number of samples to use for the monte carlo prediction)")

        super().__init__(*args, **kwargs)

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.optimizer.update_prior_lambda()

    def batch_step(self, inputs, targets, dataset_size):
        """Perform the training of a single sample of the batch
        """
        def closure():
            # Closure for the optimizer sending the loss to the optimizer
            self.optimizer.zero_grad()
            output = self.model.forward(inputs).to(self.device)
            loss = self.criterion(output.to(self.device),
                                  targets.to(self.device),
                                  reduction=self.reduction)
            return loss
        self.model.train()
        ### LOSS ###
        self.loss = self.optimizer.step(
            input_size=dataset_size, closure=closure)

    def test(self, inputs, labels):
        """Test the model on the given inputs and labels

        Args:
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels

        Returns:
            torch.Tensor: Predictions
        """
        with torch.no_grad():
            noise = []
            for _ in range(self.test_mcmc_samples):
                noise.append(torch.bernoulli(
                    torch.sigmoid(2*self.optimizer.state["lambda"])))
            if len(noise) == 0:
                noise.append(torch.where(self.optimizer.state['mu'] <= 0, torch.zeros_like(
                    self.optimizer.state['mu']), torch.ones_like(self.optimizer.state['mu'])))
            predictions = self.monte_carlo_prediction(inputs, noise)
            idx_pred = torch.argmax(predictions, dim=1)
        return len(torch.where(idx_pred == labels)[0]) / len(labels)

    def monte_carlo_prediction(self, inputs, noise):
        """Perform a monte carlo prediction on the given inputs

        Args:
            inputs (torch.Tensor): Input data
            noise (torch.Tensor, optional): Noise to use for the prediction. Defaults to None.

        Returns:
            torch.Tensor: Predictions
        """
        # Retrieve the parameters
        parameters = self.optimizer.param_groups[0]['params']
        predictions = []
        # We iterate over the parameters
        for n in noise:
            # 2p - 1
            torch.nn.utils.vector_to_parameters(2*n-1, parameters)
            prediction = self.model.forward(inputs).to(self.device)
            predictions.append(prediction)
        predictions = torch.stack(predictions, dim=2).to(self.device)
        return torch.mean(predictions, dim=2).to(self.device)

    def epoch_step(self, train_dataset, test_loader=None):
        """Perform the training of a single epoch

        Args: 
            train_dataset (torch.Tensor): Training data
            test_loader (torch.Tensor, optional): Testing data. Defaults to None.
        """
        ### SEND BATCH ###

        dataset_size = len(train_dataset) * train_dataset.batch_size
        for inputs, targets in train_dataset:
            if len(inputs.shape) == 4:
                # remove all dimensions of size 1
                inputs = inputs.squeeze()
            self.batch_step(inputs.to(self.device),
                            targets.to(self.device), dataset_size)

        ### SCHEDULER ###
        if "scheduler" in dir(self):
            self.scheduler.step()

        ### EVALUATE ###
        if test_loader is not None:
            test = []
            for testset in test_loader:
                batch = []
                for inputs, targets in testset:
                    if len(inputs.shape) == 4:
                        # remove all dimensions of size 1
                        inputs = inputs.squeeze()
                    if "test_permutations" in dir(self):
                        self.testing_accuracy.append(
                            self.test_continual(inputs.to(self.device), targets.to(self.device)))
                    else:
                        batch.append(
                            self.test(inputs.to(self.device), targets.to(self.device)))
                test.append(torch.mean(torch.tensor(batch)))
            if "test_permutations" not in dir(self):
                self.testing_accuracy.append(test)

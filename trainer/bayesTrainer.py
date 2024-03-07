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

    def __init__(self, test_mcmc_samples=1, *args, **kwargs):
        if "test_mcmc_samples" in kwargs:
            self.test_mcmc_samples = kwargs["test_mcmc_samples"]
        elif "training_parameters" in kwargs and "test_mcmc_samples" in kwargs["training_parameters"]:
            self.test_mcmc_samples = kwargs["training_parameters"]["test_mcmc_samples"]
        else:
            self.test_mcmc_samples = test_mcmc_samples
        # add the test_mcmc_samples to kwargs
        kwargs["test_mcmc_samples"] = self.test_mcmc_samples
        super().__init__(*args, **kwargs)

    def batch_step(self, inputs, targets, dataset_size):
        """Perform the training of a single sample of the batch
        """
        def closure():
            # Closure for the optimizer sending the loss to the optimizer
            self.optimizer.zero_grad()
            output = self.model.forward(inputs).to(self.device)
            loss = self.criterion(output.to(self.device),
                                  targets.to(self.device))
            return loss
        ### LOSS ###
        self.loss = self.optimizer.step(
            input_size=dataset_size, closure=closure)

    @torch.no_grad()
    def predict(self, inputs, n_samples=1):
        """Predict the output of the model on the given inputs

        Args:
            inputs (torch.Tensor): Input data

        Returns:
            torch.Tensor: Mean of the predictions
            torch.Tensor: Standard deviation of the predictions
        """
        self.model.eval()
        noise = []
        # Sample from the bernoulli distribution with p = sigmoid(2*lambda)
        for _ in range(n_samples):
            noise.append(torch.bernoulli(
                torch.sigmoid(2*self.optimizer.state["lambda"])))
        if len(noise) == 0:
            noise.append(torch.where(self.optimizer.state['mu'] <= 0,
                                     torch.zeros_like(
                self.optimizer.state['mu']),
                torch.ones_like(self.optimizer.state['mu'])))
        # Retrieve the parameters of the networks
        parameters = [p for p in self.optimizer.param_groups[0]
                      ['params'] if p.requires_grad]
        predictions = []
        # We iterate over the parameters
        for n in noise:
            # Sample neural networks weights
            torch.nn.utils.vector_to_parameters(2*n-1, parameters)
            # Predict with this sampled network
            prediction = self.model.forward(inputs.to(self.device))
            predictions.append(prediction)
        return predictions

    @torch.no_grad()
    def test(self, inputs, labels):
        """Test the model on the given inputs and labels

        Args:
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels

        Returns:
            torch.Tensor: Predictions
        """
        self.model.eval()
        predictions = self.predict(inputs, n_samples=self.test_mcmc_samples)
        predictions = torch.stack(predictions, dim=0)
        predictions = torch.mean(predictions, dim=0)
        if self.model.output_function == "sigmoid":
            # apply exponential to get the probability
            predictions = torch.where(predictions >= 0.5, torch.ones_like(
                predictions), torch.zeros_like(predictions))
        else:
            predictions = torch.argmax(predictions, dim=1)
        return torch.mean((predictions == labels).float())

    def epoch_step(self, train_dataset, test_loader=None):
        """Perform the training of a single epoch

        Args: 
            train_dataset (torch.Tensor): Training data
            test_loader (torch.Tensor, optional): Testing data. Defaults to None.
        """
        ### SEND BATCH ###
        dataset_size = len(train_dataset) * train_dataset.batch_size
        self.model.train()
        for inputs, targets in train_dataset:
            self.batch_step(inputs.to(self.device),
                            targets.to(self.device), dataset_size)

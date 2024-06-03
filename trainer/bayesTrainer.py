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

    def batch_step(self, inputs, targets):
        """Perform the training of a single sample of the batch
        """
        def closure():
            # Closure for the optimizer sending the loss to the optimizer
            self.optimizer.zero_grad()
            forward = self.model.forward(inputs).to(self.device)
            if self.label_trick is not None and self.label_trick:
                unique_labels, trick_targets = self.label_trick(targets)
                loss = self.criterion(
                    forward[:, unique_labels].to(self.device),
                    trick_targets.to(self.device),
                    reduction='sum'
                )
            else:
                loss = self.criterion(forward, targets.to(
                    self.device, reduction='sum'))
            return loss
        ### LOSS ###
        self.loss = self.optimizer.step(closure=closure)

    def predict(self, inputs, n_samples=1):
        """Predict the output of the model on the given inputs

        Args:
            inputs (torch.Tensor): Input data

        Returns:
            torch.Tensor: Mean of the predictions
            torch.Tensor: Standard deviation of the predictions
        """
        self.model.eval()
        with torch.no_grad():
            noise = []
            # Sample from the bernoulli distribution with p = sigmoid(2*lambda)
            if n_samples > 0:
                for _ in range(n_samples):
                    network_sample = torch.bernoulli(
                        torch.sigmoid(2*self.optimizer.state["lambda"]))
                    noise.append(2*network_sample-1)
            elif n_samples == 0:
                # If we do not want to sample, we do a bernouilli
                noise.append(2*torch.where(
                    self.optimizer.state["lambda"] <= 0,
                    torch.zeros_like(self.optimizer.state["lambda"]),
                    torch.ones_like(self.optimizer.state["lambda"])
                )-1)
            else:
                # If we only take lambda as the weights
                noise.append(self.optimizer.state["lambda"])

            # Retrieve the parameters of the networks
            parameters = [p for p in self.optimizer.param_groups[0]
                          ['params'] if p.requires_grad]
            predictions = []
            # We iterate over the parameters
            for n in noise:
                # Sample neural networks weights
                torch.nn.utils.vector_to_parameters(n, parameters)
                # Predict with this sampled network
                prediction = self.model.forward(inputs.to(self.device))
                predictions.append(prediction)
        return predictions

    def test(self, inputs, labels):
        """Test the model on the given inputs and labels

        Args:
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels

        Returns:
            torch.Tensor: Predictions
        """
        predictions = self.predict(inputs, n_samples=self.test_mcmc_samples)
        predictions = torch.stack(predictions, dim=0)
        if self.model.output_function == "sigmoid":
            # apply exponential to get the probability
            predicted = torch.where(torch.mean(predictions, dim=0) >= 0.5, torch.ones_like(
                predictions), torch.zeros_like(predictions))
        else:
            predicted = torch.argmax(torch.mean(predictions, dim=0), dim=1)
        return torch.mean((predicted == labels).float()), predictions

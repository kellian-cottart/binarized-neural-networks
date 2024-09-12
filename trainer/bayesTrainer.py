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

    def __init__(self, model, optimizer, optimizer_parameters, criterion, reduction, device, output_function, *args, **kwargs):
        optimizer_parameters["model"] = model
        self.optimizer = optimizer(**optimizer_parameters)
        super(BayesTrainer, self).__init__(
            model, optimizer, optimizer_parameters, criterion, reduction, device, output_function, *args, **kwargs)

    def batch_step(self, inputs, targets):
        """Perform the training of a single sample of the batch
        """
        def closure():
            # Closure for the optimizer sending the loss to the optimizer
            self.optimizer.zero_grad()
            forward = self.output_apply(
                self.model.forward(inputs).to(self.device))
            loss = self.criterion(
                forward,
                targets.to(self.device),
                reduction=self.reduction)
            return loss, forward
        ### LOSS ###
        self.loss = self.optimizer.step(closure=closure)

    def predict(self, inputs):
        """Predict the output of the model on the given inputs

        Args:
            inputs (torch.Tensor): Input data

        Returns:
            torch.Tensor: Mean of the predictions
            torch.Tensor: Standard deviation of the predictions
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.optimizer.get_mc_predictions(forward_function=self.model.forward,
                                                            inputs=inputs,
                                                            ret_numpy=False)
            predictions = torch.stack(predictions)
            predictions = self.output_apply(predictions)
        return predictions

    def test(self, inputs, labels):
        """Test the model on the given inputs and labels

        Args:
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels

        Returns:
            torch.Tensor: Predictions
        """
        predictions = self.predict(inputs)
        if self.output_function == "sigmoid":
            # apply exponential to get the probability
            predicted = torch.where(torch.mean(predictions, dim=0) >= 0.5, torch.ones_like(
                predictions), torch.zeros_like(predictions))
        else:
            predicted = torch.argmax(torch.mean(predictions, dim=0), dim=1)
        return torch.mean((predicted == labels).float()), predictions

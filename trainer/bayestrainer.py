from .trainer import Trainer
import torch


class BayesTrainer(Trainer):
    """Extended Trainer class to cover the special case of BayesBiNN

    Necessity to have a different training function to implement mu and lambda properly 

    Args:
        Trainer (Trainer): Trainer class to extend
        *args: Variable length argument list (for the Trainer class)
        **kwargs: Arbitrary keyword arguments (most likely optimizer or scheduler parameters)
    """

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.optimizer.update_prior_lambda()

    def batch_step(self, inputs, targets):
        """Perform the training of a single sample of the batch
        """
        def closure():
            # Closure for the optimizer sending the loss to the optimizer
            self.optimizer.zero_grad()
            output = self.model.forward(inputs)
            loss = self.criterion(output, targets)
            return loss

        ### FORWARD PASS ###
        inputs = inputs.view(inputs.shape[0], -1).to(self.device)
        targets = targets.to(self.device)
        prediction = self.model.forward(inputs)

        ### LOSS ###
        self.loss = self.criterion(prediction, targets)

        ### BACKWARD PASS ###
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step(closure=closure)

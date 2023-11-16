from .trainer import Trainer
import torch


class BayesTrainer(Trainer):
    """Extended Trainer class to cover the special case of BayesBiNN

    Necessity to have a different training function to implement mu and lambda properly 
    """

    def __init__(self, model, optimizer, optimizer_parameters, criterion, device, *args, **kwargs):
        self.model = model
        self.optimizer = optimizer(model.parameters(), **optimizer_parameters)
        self.criterion = criterion
        self.device = device
        self.training_accuracy = []
        self.testing_accuracy = []

    def epoch_step(self, train_loader, test_loader=None):
        super().epoch_step(train_loader, test_loader)

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.optimizer.update_prior_lambda(
            self.optimizer.lambda_)

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

from .trainer import Trainer
import torch


class BayesTrainer(Trainer):
    """Extended Trainer class to cover the special case of BayesBiNN

    Necessity to have a different training function to implement mu and lambda properly 
    """

    def __init__(self, model, optimizer, optimizer_parameters, criterion, device, *args, **kwargs):
        self.model = model
        self.optimizer = optimizer(model, **optimizer_parameters)
        self.criterion = criterion
        self.device = device
        self.training_accuracy = []
        self.testing_accuracy = []

    def epoch_step(self, train_loader, test_loader=None):
        super().epoch_step(train_loader, test_loader)

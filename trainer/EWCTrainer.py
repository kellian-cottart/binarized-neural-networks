import torch
import tqdm
import os
from .gpuTrainer import GPUTrainer


class EWCTrainer(GPUTrainer):
    """Trainer that does not require the usage of DataLoaders

    Args:
        model (torch.nn.Module): Model to train
        optimizer (torch.optim): Optimizer to use
        optimizer_parameters (dict): Parameters of the optimizer
        criterion (torch.nn): Loss function
        reduction (str): Reduction method of the loss
        device (torch.device): Device to use for the training
        kwargs: Additional arguments
            scheduler (torch.optim.lr_scheduler, optional): Scheduler to use. Defaults to None.
            scheduler_parameters (dict, optional): Parameters of the scheduler. Defaults to None.
    """

    def __init__(self, importance=1, *args, **kwargs):
        self.importance = importance
        self.fisher = []  # list of diagonal fisher information matrices per task
        super(EWCTrainer, self).__init__(*args, **kwargs)

    def compute_diag_fisher(self):
        pass

    def ewc_loss(self):
        # regularization is equal to the sum over parameters of the product between
        # diagonal fisher information matrix at param i and the difference between the
        # current weights and the weights at the end of task previous task for param i

        return self.importance/2

    def batch_step(self, inputs, targets):
        """Perform the training of a single batch

        Args:
            inputs (torch.Tensor): Input data
            targets (torch.Tensor): Labels
        """
        ### LOSS ###
        self.model.train()
        forward = self.model.forward(inputs).to(self.device)
        forward, targets = self.label_trick_loss(forward, targets)
        forward = self.output_apply(forward)
        if forward.dim() == 2:
            forward = forward.unsqueeze(0)
        forward = forward.mean(dim=0)
        self.loss = self.criterion(
            forward.to(self.device),
            targets.to(self.device),
            reduction=self.reduction,
        ) + self.ewc_loss()
        ### BACKWARD PASS ###
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

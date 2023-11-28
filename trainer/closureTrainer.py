from .gpuTrainer import GPUTrainer


class ClosureTrainer(GPUTrainer):
    """Trainer but with a closure for the optimizer

    Args:
        Trainer (Trainer): Trainer class to extend
        *args: Variable length argument list (for the Trainer class)
        **kwargs: Arbitrary keyword arguments (most likely optimizer or scheduler parameters)
    """

    def batch_step(self, inputs, targets):
        """Perform the training of a single sample of the batch
        """
        def closure():
            # Closure for the optimizer sending the loss to the optimizer
            self.optimizer.zero_grad()
            output = self.model.forward(inputs)
            loss = self.criterion(output, targets)
            return loss

        self.model.train()
        ### LOSS ###
        self.loss = self.criterion(
            self.model.forward(inputs),
            targets,
            reduction=self.reduction)

        ### BACKWARD PASS ###
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step(closure)

import torch
from copy import deepcopy
from dataloader import *
from optimizers import MetaplasticAdam


class GPUTrainer:
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

    def __init__(self, model, optimizer, optimizer_parameters, criterion, reduction, device, output_function, task=None, n_tasks=None, *args, **kwargs):
        self.model = model
        if not hasattr(self, "optimizer"):
            self.optimizer = optimizer(
                self.model.parameters(),
                **optimizer_parameters
            )
        self.criterion = criterion
        self.device = device
        self.task = task
        self.n_tasks = n_tasks
        self.training_accuracy = []
        self.testing_accuracy = []
        self.mean_testing_accuracy = []
        self.reduction = reduction
        self.output_function = output_function
        self.hessian = []
        self.gradient = []
        if "regularizer" in kwargs and kwargs["regularizer"]["type"].lower() == "ewc":
            self.ewc = True
            self.lbda = kwargs["regularizer"]["lambda"]
            self.means = {}
            self.precision_matrices = {}
            self.old_tasks = []
            self.params = {
                n: p for n, p in self.model.named_parameters() if p.requires_grad}
            self.update_ewc()
        # Label trick
        if "label_trick" in kwargs:
            self.label_trick = kwargs["label_trick"]
        # Scheduler addition
        if "scheduler" in kwargs:
            scheduler = kwargs["scheduler"]
            scheduler_parameters = kwargs["scheduler_parameters"]
            self.scheduler = scheduler(
                self.optimizer, **scheduler_parameters)

    def reset_optimizer(self, optimizer_parameters):
        """Reset the optimizer parameters such as momentum and learning rate

        Args:
            optimizer_parameters (dict): Parameters of the optimizer
        """
        self.optimizer = self.optimizer.__class__(
            self.model.parameters(), **optimizer_parameters)

    def label_trick_loss(self, inputs, targets):
        """ When dealing with CIL, we need to not update the synapses linked to neurons that are not used in the task
        Args:
            inputs (torch.Tensor): Inputs (forward pass with no activation)
            targets (torch.Tensor): Targets (labels)
        """
        if not hasattr(self, "label_trick") or self.label_trick == False:
            return inputs, targets
        # Select unique labels in targets
        unique_labels = torch.unique(targets).sort()[0]
        # We update synapses only for the output neurons that are used in the task
        new_targets = targets.clone()
        for i, label in enumerate(unique_labels):
            new_targets[targets == label] = i
        if inputs.dim() == 2:
            new_inputs = inputs[:, unique_labels]
        else:
            new_inputs = inputs[:, :, unique_labels]
        return new_inputs, new_targets

    def output_apply(self, forward):
        if self.output_function == "sigmoid":
            return torch.sigmoid(forward)
        elif self.output_function == "softmax":
            return torch.nn.functional.softmax(forward, dim=1 if forward.dim() == 2 else 2)
        elif self.output_function == "log_softmax":
            return torch.nn.functional.log_softmax(forward, dim=1 if forward.dim() == 2 else 2)
        else:
            return forward

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
        )
        if hasattr(self, "ewc") and self.ewc == True:
            self.loss += self.lbda * self.ewc_loss()
        self.old_tasks = []
        ### BACKWARD PASS ###
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def compute_hessian(self):
        # only for last layer
        layer = self.model.layers[-2].lambda_
        diagonal = torch.zeros_like(layer)
        grad = torch.autograd.grad(
            self.loss, layer, create_graph=True)[0]
        for i in range(len(grad)):
            for j in range(len(grad[i])):
                # autograd to compute the hessian diagonal
                hessian = torch.autograd.grad(
                    grad[i][j], layer, create_graph=True)[0]
                diagonal[i][j] = hessian[i][j]
        return diagonal

    def evaluate(self, test_loader, batch_size=1024, train_loader=None, batch_params=None, permutations=None):
        """ Evaluate the model on the test sets

        Args:
            test_loader (iterable): Testing data
            train_loader (iterable, optional): Training data. Defaults to None.
            batch_params (dict, optional): Parameters of the batch. Defaults to None.

        Returns:
            float: mean accuracy on the test sets
        """
        test_predictions = []
        labels = []
        test = []
        ### TESTING SET ###
        for i, dataset in enumerate(test_loader):
            if batch_params is not None:
                self.model.load_bn_states(batch_params[i])
            batch = []
            target_batch = []
            n_batches = len(dataset) // batch_size + \
                (len(dataset) % batch_size != 0)
            for i in range(n_batches):
                inputs, targets = dataset.__getbatch__(
                    i*batch_size, batch_size)
                accuracy, predictions = self.test(
                    inputs.to(self.device), targets.to(self.device))
                batch.append(accuracy)
                test_predictions.append(predictions)
                target_batch.append(targets)
            labels.append(torch.cat(target_batch))
            test.append(torch.mean(torch.tensor(batch)))
        test = torch.tensor(test)
        self.testing_accuracy.append(test)
        self.mean_testing_accuracy.append(test.mean())
        # ### TRAINING SET ###
        # # Conditional, we only compute the training accuracy if the training data is provided
        # if train_loader is not None:
        #     train = []
        #     for i, dataloader in enumerate(train_loader):
        #         if batch_params is not None:
        #             self.model.load_bn_states(batch_params[i])
        #         batch = []
        #         for inputs, targets in dataloader:
        #             accuracy, predictions = self.test(inputs, targets)
        #             batch.append(accuracy)
        #         train.append(torch.mean(torch.tensor(batch)))
        #     self.training_accuracy.append(train)
        # # Return the prediction, and the corresponding labels.
        return test_predictions, labels

    def predict(self, inputs):
        """Predict the labels of the given inputs

        Args:
            inputs (torch.Tensor): Input data

        Returns:
            torch.Tensor: Predictions
        """
        self.model.eval()
        with torch.no_grad():
            # Specifying backwards=False in case the forward and the backward are different
            predictions = self.output_apply(self.model.forward(
                inputs.to(self.device), backwards=False))
        return predictions

    def test(self, inputs, labels):
        """ Predict labels for a full dataset and retrieve accuracy

        Args:
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Labels of the data

        Returns:
            float: Accuracy of the model on the dataset
            torch.Tensor: Predictions

        """
        ### ACCURACY COMPUTATION ###
        full_predictions = self.predict(inputs)
        if len(full_predictions.shape) < 3:
            full_predictions = full_predictions.unsqueeze(0)
        predictions = torch.mean(full_predictions, dim=0)
        if self.output_function == "sigmoid":
            # apply exponential to get the probability
            predicted = torch.where(predictions >= 0.5, torch.ones_like(
                predictions), torch.zeros_like(predictions))
        else:
            predicted = torch.argmax(predictions, dim=1)
        return torch.mean((predicted == labels.to(self.device)).float()), full_predictions

    def pbar_update(self, pbar, epoch, n_epochs, task_id, n_tasks):
        """Update the progress bar with the current loss and accuracy"""
        pbar.set_description(
            f"Epoch {epoch+1}/{n_epochs} - Task {task_id+1}/{n_tasks}")
        # creation of a dictionnary with the name of the test set and the accuracy
        kwargs = {}
        if len(self.testing_accuracy) > 0:
            if "training_accuracy" in dir(self) and len(self.training_accuracy) > 0:
                kwargs = {
                    f"task {i+1}": f"Test: {test_acc:.2%} - Train: {train_acc:.2%}" for i, test_acc, train_acc in zip(range(len(self.testing_accuracy[-1])), self.testing_accuracy[-1], self.training_accuracy[-1])
                }
            else:
                kwargs = {
                    f"task {i+1}": f"Test: {accuracy:.2%}" for i, accuracy in enumerate(self.testing_accuracy[-1])
                }
            pbar.set_postfix(loss=self.loss.item())
            # Do a pretty print of our results
            pbar.write("==================================")
            pbar.write("Accuracies: ")
            for key, value in kwargs.items():
                pbar.write(f"\t{key}: {value}")

    def save(self, path):
        """Save the model
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load the model
        """
        self.model.load_state_dict(torch.load(path))

    def evaluate_tasks(self, dataset, task, permutations, batch_size=128, train_dataset=None, batch_params=None):
        if "Permuted" in task:
            dataset = dataset[0]
            predictions, labels = self.evaluate(
                test_permuted_dataset(
                    test_dataset=dataset,
                    permutations=permutations
                ),
                batch_size=dataset.data.shape[0],
                batch_params=batch_params)
        else:
            predictions, labels = self.evaluate(
                dataset,
                batch_size=batch_size,
                batch_params=batch_params)
        return predictions, labels

    def epoch_step(self, batch_size, test_batch_size, task_train_dataset, test_dataset, task_id, permutations, epoch, pbar, epochs, continual=False, batch_params=None):
        num_batches = len(task_train_dataset) // batch_size + 1
        task_train_dataset.shuffle()
        for n_batch in range(num_batches):
            ### TRAINING ###
            batch, labels = batch_yielder(
                dataset=task_train_dataset,
                task=self.task,
                batch_size=batch_size,
                task_id=task_id,
                iteration=n_batch,
                max_iterations=epochs*num_batches,
                permutations=permutations,
                epoch=epoch,
                continual=continual
            )
            self.batch_step(batch, labels)
        if self.optimizer in [MetaplasticAdam] and self.model.affine:
            batch_params[task_id] = self.model.save_bn_states()
        ### TESTING ###
        # Depending on the task, we also need to use the framework on the test set and show training or not
        predictions, labels = self.evaluate_tasks(
            dataset=test_dataset,
            task=self.task,
            permutations=permutations,
            batch_size=test_batch_size,
            train_dataset=task_train_dataset,
            batch_params=batch_params if self.optimizer in [
                MetaplasticAdam] and self.model.affine else None
        )
        self.pbar_update(
            pbar, epoch, n_epochs=epochs, n_tasks=self.n_tasks, task_id=task_id)
        if self.optimizer in [MetaplasticAdam] and self.model.affine:
            self.model.load_bn_states(batch_params[task_id])

        ### UPDATING EWC ###
        if hasattr(self, "ewc") and self.ewc == True and epoch == epochs-1:
            self.old_tasks.append(batch)
            self.update_ewc()
        return predictions, labels

    def update_ewc(self):
        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in deepcopy(self.params).items():
            self.means[n] = p.clone().detach()
        self.compute_diag_fisher()

    def ewc_loss(self):
        loss = 0
        for n, p in self.params.items():
            loss += (self.precision_matrices[n]
                     * (p - self.means[n]) ** 2).sum()
        return loss

    def compute_diag_fisher(self):
        fisher = {}
        for n, p in self.params.items():
            fisher[n] = torch.zeros_like(p)
        if len(self.old_tasks) > 0:
            for task in self.old_tasks:
                self.optimizer.zero_grad()
                forward = self.model.forward(task)
                forward = self.output_apply(forward)
                targets = forward.max(1)[1]
                self.criterion(forward, targets).backward()
                for n, p in self.params.items():
                    fisher[n] += p.grad.data ** 2 / len(self.old_tasks)
        self.precision_matrices = {n: p for n, p in fisher.items()}

import torch
from .structures import *


def permuted_dataset(dataset, batch_size, continual, task_id, iteration, max_iterations, permutations, epoch):
    perm = permutations[task_id]
    batch_data, targets = dataset.__getbatch__(
        batch_size * iteration, batch_size)
    shape = batch_data.shape
    batch_data = batch_data.to(perm.device).view(shape[0], shape[1], -1)
    split = 0.75
    n_images_taken_b1 = int((1 - (iteration*(epoch+1) - int(max_iterations*split)) / (
        max_iterations - int(max_iterations*split))) * batch_size)  # Ratio between the number of images taken from batch perm 1 and batch perm 2
    if len(permutations) > task_id + 1 and batch_size - n_images_taken_b1 > 0 and continual == True:
        next_perm = permutations[task_id + 1]
        batch_data = torch.cat(
            (batch_data[:n_images_taken_b1, :, perm],
                batch_data[n_images_taken_b1:, :, next_perm]), dim=0).view(shape)
    else:
        batch_data = batch_data[:, :, perm].view(shape)
    return batch_data, targets


def batch_yielder(dataset, task, batch_size=128, continual=None, task_id=None, iteration=None, max_iterations=None, permutations=None, epoch=None):
    batch_data, targets = None, None
    if "Permuted" in task:
        batch_data, targets = permuted_dataset(dataset, batch_size, continual,
                                               task_id, iteration, max_iterations, permutations, epoch)
    else:
        batch_data, targets = dataset.__getbatch__(
            batch_size * iteration, batch_size)
    return batch_data.to(dataset.device), targets.to(dataset.device)


def test_permuted_dataset(test_dataset, permutations):
    for i in range(len(permutations)):
        data, targets = permuted_dataset(dataset=test_dataset, batch_size=test_dataset.data.shape[
                                         0], continual=False, task_id=i, iteration=0, max_iterations=1, permutations=permutations, epoch=0)
        yield GPUTensorDataset(data, targets, device=test_dataset.device)


def evaluate_tasks(dataset, task, net_trainer, permutations, batch_size=1024, train_dataset=None, batch_params=None):
    if "Permuted" in task:
        dataset = dataset[0]
        predictions, labels = net_trainer.evaluate(
            test_permuted_dataset(
                test_dataset=dataset,
                permutations=permutations
            ),
            batch_size=dataset.data.shape[0],
            batch_params=batch_params)
    else:
        predictions, labels = net_trainer.evaluate(
            dataset,
            batch_size=batch_size,
            batch_params=batch_params)
    return predictions, labels

import torch


def permuted_dataset(train_dataset, batch_size, continual, task_id, iteration, max_iterations, permutations, epoch):
    train_dataset.data = train_dataset.data.to(permutations[0].device)
    train_dataset.targets = train_dataset.targets.to(permutations[0].device)
    perm = permutations[task_id]
    split = 0.75
    shape = train_dataset.data.shape[-2]*train_dataset.data.shape[-1]
    iteration_on_total = iteration*(epoch+1)
    if len(permutations) > task_id + 1 and iteration_on_total >= int(max_iterations*split) and continual == True:
        next_perm = permutations[task_id + 1]
        n_images_taken = int((1 - (iteration_on_total - int(max_iterations*split)) / (
            max_iterations - int(max_iterations*split))) * batch_size)
        batch_data = train_dataset.data[batch_size * iteration: batch_size *
                                        iteration + n_images_taken].view(-1, shape)[:, perm]
        next_data = train_dataset.data[batch_size * iteration + n_images_taken: batch_size * (
            iteration + 1)].view(-1, shape)[:, next_perm]
        batch_data = torch.cat((batch_data, next_data))
    else:
        batch_data = train_dataset.data[batch_size * iteration: batch_size * (
            iteration + 1)].view(-1, shape)[:, perm]
    targets = train_dataset.targets[batch_size *
                                    iteration: batch_size * (iteration + 1)]
    batch_data = batch_data.view(batch_data.shape[0],
                                 train_dataset.data.shape[-3],
                                 train_dataset.data.shape[-2],
                                 train_dataset.data.shape[-1])
    return batch_data, targets


def class_incremental_dataset(train_dataset, batch_size, iteration, permutations, task_id):
    """ Returns a batch of data and targets for the class incremental learning scenario

    Args:
        train_dataset (torch.utils.data.Dataset): The dataset to sample from
        classes (tensor): The classes to sample from the dataset
    """
    train_dataset.data = train_dataset.data.to(permutations[0].device)
    train_dataset.targets = train_dataset.targets.to(permutations[0].device)
    classes = permutations[task_id]
    indexes = torch.isin(train_dataset.targets, classes)
    shape = torch.prod(torch.tensor(train_dataset.data.shape[1:]))
    batch_data = train_dataset.data[indexes][batch_size * iteration: batch_size * (
        iteration + 1)].view(-1, shape)
    targets = train_dataset.targets[indexes][batch_size *
                                             iteration: batch_size * (iteration + 1)]
    batch_data = batch_data.view(batch_data.shape[0],
                                 train_dataset.data.shape[-3],
                                 train_dataset.data.shape[-2],
                                 train_dataset.data.shape[-1])
    return batch_data, targets


def stream_dataset(train_dataset, iteration, n_tasks, current_task, batch_size=1):
    """ Returns a batch of data and targets for the stream learning scenario,
    not including the rest of the unused data as in drop last

    Args:
        train_dataset (torch.utils.data.Dataset): The dataset to sample from
        iteration (int): The current iteration
        n_tasks (int): The number of subsets to split the dataset in
        current_task (int): The current task to sample from
    """
    n_samples = train_dataset.data.shape[0] // n_tasks
    start = n_samples * current_task
    end = n_samples * (current_task + 1)
    batch_data = train_dataset.data[start:end][batch_size *
                                               iteration: batch_size * (iteration + 1)]
    targets = train_dataset.targets[start:end][batch_size *
                                               iteration: batch_size * (iteration + 1)]
    batch_data = batch_data.view(batch_data.shape[0],
                                 train_dataset.data.shape[-3],
                                 train_dataset.data.shape[-2],
                                 train_dataset.data.shape[-1])
    return batch_data, targets


def special_task_selector(data, train_dataset, batch_size=128, continual=None, task_id=None, iteration=None, max_iterations=None, permutations=None, epoch=None):
    batch_data, targets = None, None
    if "Permuted" in data["task"]:
        batch_data, targets = permuted_dataset(train_dataset, batch_size, continual,
                                               task_id, iteration, max_iterations, permutations, epoch)
    elif "CIL" in data["task"]:
        batch_data, targets = class_incremental_dataset(
            train_dataset,
            batch_size=batch_size,
            permutations=permutations,
            task_id=task_id,
            iteration=iteration)

    elif "Stream" in data["task"]:
        batch_data, targets = stream_dataset(
            train_dataset,
            iteration=iteration,
            n_tasks=data["n_tasks"],
            current_task=task_id,
            batch_size=batch_size)
    else:
        # yield a standard batch
        batch_data = train_dataset.data[batch_size *
                                        iteration: batch_size * (iteration + 1)]
        targets = train_dataset.targets[batch_size *
                                        iteration: batch_size * (iteration + 1)]
        batch_data = batch_data.view(batch_size,
                                     train_dataset.data.shape[-3],
                                     train_dataset.data.shape[-2],
                                     train_dataset.data.shape[-1])
    return batch_data.to(train_dataset.device), targets.to(train_dataset.device)


def test_permuted_dataset(test_dataset, permutations):
    for i in range(len(permutations)):
        yield [permuted_dataset(train_dataset=test_dataset, batch_size=test_dataset.data.shape[0], continual=False, task_id=i, iteration=0, max_iterations=1, permutations=permutations, epoch=0)]


def test_class_incremental_dataset(test_dataset, permutations):
    for i in range(len(permutations)):
        yield [class_incremental_dataset(train_dataset=test_dataset, batch_size=test_dataset.data.shape[0]//len(permutations), iteration=0, permutations=permutations, task_id=i)]


def iterable_evaluation_selector(data, test_dataset, net_trainer, permutations, batch_size=512, train_dataset=None, batch_params=None):
    name_loader = None
    if "Permuted" in data["task"]:
        predictions, labels = net_trainer.evaluate(
            test_permuted_dataset(
                test_dataset=test_dataset,
                permutations=permutations
            ),
            batch_params=batch_params)
    elif "CIL" in data["task"]:
        predictions, labels = net_trainer.evaluate(
            test_class_incremental_dataset(
                test_dataset=test_dataset,
                permutations=permutations
            ),
            batch_params=batch_params)
    else:
        num_batches = len(test_dataset) // batch_size
        loader = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_data = test_dataset.data[start_idx:end_idx]
            targets = test_dataset.targets[start_idx:end_idx]
            batch_data = batch_data.view(batch_size,
                                         train_dataset.data.shape[-3],
                                         train_dataset.data.shape[-2],
                                         train_dataset.data.shape[-1])
            loader.append((batch_data, targets))
        predictions, labels = net_trainer.evaluate(
            [loader],
            batch_params=batch_params)

        # Unpack the list of lists into just a list

    return name_loader, predictions, labels

import torch


def iterable_selector(data, train_loader, shape, target_size):
    task_iterator = None
    if "Permuted" in data["task"]:
        # Create n_tasks permutations of the datasetc
        permutations = [torch.randperm(torch.prod(torch.tensor(shape)))
                        for _ in range(data["n_tasks"])]
        task_iterator = train_loader[0].permute_dataset(permutations)
    elif "CIL" in data["task"]:
        # Create n_tasks subsets of n_classes (need to permute the selection of classes for each task)
        rand = torch.randperm(target_size)
        # n_tasks subsets of n_classes without overlapping
        permutations = [rand[i:i+data["n_classes"]]
                        for i in range(0, target_size, data["n_classes"])]
        task_iterator = train_loader[0].class_incremental_dataset(
            permutations=permutations)
    elif "Stream" in data["task"]:
        # Split the dataset in n_tasks subsets
        task_iterator = train_loader[0].stream_dataset(
            data["n_subsets"])
    else:
        task_iterator = train_loader
    return task_iterator, permutations


def iterable_evaluation_selector(data, train_loader, test_loader, net_trainer, permutations, batch_params=None):
    name_loader = None
    if "Permuted" in data["task"]:
        predictions, labels = net_trainer.evaluate(test_loader[0].permute_dataset(
            permutations), train_loader=train_loader[0].permute_dataset(permutations) if "show_train" in data and data["show_train"] else None, batch_params=batch_params)
    elif "CIL" in data["task"]:
        predictions, labels = net_trainer.evaluate(test_loader[0].class_incremental_dataset(
            permutations=permutations), train_loader=train_loader[0].class_incremental_dataset(permutations) if "show_train" in data and data["show_train"] else None, batch_params=batch_params)
        name_loader = [f"Classes {i}-{i+data['n_classes']-1}" for i in range(
            0, data["n_tasks"]*data["n_classes"], data["n_classes"])]
    else:
        predictions, labels = net_trainer.evaluate(
            test_loader, train_loader=train_loader if "show_train" in data and data["show_train"] else None, batch_params=batch_params)

    return name_loader, predictions, labels

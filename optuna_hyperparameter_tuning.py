from dataloader import *
from optimizers import *
import trainer
import models
import torch
import os
import optuna
import json
from models.layers.activation import Sign
import argparse
from utils.iterable import *

### ARGUMENTS ###
parser = argparse.ArgumentParser(description="Gridsearch for the BiNN")
parser.add_argument(
    "--device", type=str, default="0", help="Device to use (0, 1, 2, etc.), -1 is CPU")
parser.add_argument(
    "--n_trials", type=int, default=500, help="Number of trials")
parser.add_argument(
    "--study", type=str, default="gridsearch/study-default", help="Name of the study")
parser.add_argument(
    "--db", type=str, default="gridsearch/gridsearch-2.sqlite3", help="Name of the database")
parser.add_argument(
    "--task", type=str, default="PermutedMNIST", help="Task to perform (FrameworkDataset). Ex: PermutedMNIST, StreamFashion, CILCIFAR100")
parser.add_argument(
    "--norm", type=str, default="batchnorm", help="Normalization to use (batchnorm, layernorm, groupnorm)")
parser.add_argument(
    "--layers", type=str, default="512", help="Hidden layers of the network. Ex: 512, 1024-1024")
parser.add_argument(
    "--activation", type=str, default="relu", help="Activation function to use. Ex: relu, sign")
parser.add_argument(
    "--n_tasks", type=int, default=10, help="Number of tasks to perform")


### GENERAL CONFIGURATION ###
DEVICE = f"cuda:{parser.parse_args().device}" if parser.parse_args(
).device != "-1" else "cpu"  # Device to use
print("STARTING GRIDSEARCH ON {}".format(DEVICE))
N_TRIALS = parser.parse_args().n_trials  # Number of trials
PADDING = 2

### PATHS ###
DATASETS_PATH = "datasets"
STUDY = parser.parse_args().study  # Name of the study


def train_iteration(trial):
    """ Train a single network

    Args:
        trial (optuna.Trial): Optuna trial
    """
    ### PARAMETERS ###
    lr_max = trial.suggest_float("lr_max", 1, 100, step=0.5)
    likelihood_coeff = trial.suggest_float(
        "likelihood_coeff", 0, 50, step=0.1)
    kl_coeff = trial.suggest_float(
        "kl_coeff", 0, 10, step=0.1)

    ### LAMBDA PARAMETERS ###
    # suggest int
    n_samples_backward = trial.suggest_int("n_samples_backward", 1, 10, step=1)
    init_law = trial.suggest_categorical("init_law", ["gaussian"])
    init_param = trial.suggest_float("init_param", 0, 0.15, step=0.01)
    temperature = trial.suggest_float("temperature", 0.5, 1, step=0.1)

    ### TASK PARAMETERS ###
    seed = trial.suggest_categorical(
        "seed", [1000])
    epochs = trial.suggest_categorical(
        "epochs", [20])
    batch_size = trial.suggest_categorical(
        "batch_size", [128, 256, 512])
    task = trial.suggest_categorical(
        "task", [parser.parse_args().task])
    n_tasks = trial.suggest_categorical(
        "n_tasks", [parser.parse_args().n_tasks])
    n_classes = trial.suggest_categorical(
        "n_classes", [1])
    normalization = trial.suggest_categorical(
        "normalization", [parser.parse_args().norm])

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_default_device(DEVICE)
    torch.set_default_dtype(torch.float32)

    ### DATA TO DISPLAY IN OPTUNA ###
    data = {
        "nn_type": models.BiBayesianNN,
        "nn_parameters": {
            "layers": [int(i) for i in parser.parse_args().layers.split("-")],
            "device": DEVICE,
            "dropout": False,
            "normalization": normalization,
            "init": init_law,
            "std": init_param,
            "n_samples_forward": 1,
            "n_samples_backward": n_samples_backward,
            "tau": temperature,
            "binarized": False,
            "activation_function": torch.functional.F.relu if parser.parse_args().activation == "relu" else Sign(),
            "output_function": "log_softmax",
            "eps": 1e-5,
            "momentum": 0,
            "running_stats": False,
            "affine": False,
            "bias": False,
        },
        "criterion": torch.nn.functional.nll_loss,
        "reduction": "sum",
        "training_parameters": {
            'n_epochs': epochs,
            'batch_size': batch_size,
            'resize': True,
            'data_aug_it': 10,
            "continual": True,
        },
        "optimizer": BHUparallel,
        "optimizer_parameters": {
            "lr_max": lr_max,
            "kl_coeff": kl_coeff,
            "likelihood_coeff": likelihood_coeff,
            "normalize_gradients": False,
        },
        "task": task,
        "n_tasks": n_tasks,
        "n_classes": n_classes,
    }

    ### LOADER ###
    # Creates a GPULoading instance that loads any dataset in the same format
    loader = GPULoading(padding=PADDING,
                        device=DEVICE,
                        as_dataset=False)
    resize = data["training_parameters"]["resize"] if "resize" in data["training_parameters"] else False
    ### LOADING DATASET ###
    train_dataset, test_dataset, shape, target_size = task_selection(loader=loader,
                                                                     task=data["task"],
                                                                     n_tasks=data["n_tasks"],
                                                                     batch_size=batch_size,
                                                                     resize=resize,
                                                                     iterations=data["training_parameters"]["data_aug_it"])
    ### CREATING PERMUTATIONS ###
    permutations = None
    if "Permuted" in data["task"]:
        permutations = [torch.randperm(torch.prod(torch.tensor(shape)))
                        for _ in range(data["n_tasks"])]
    if "CIL" in data["task"]:
        # Create the permutations for the class incremental scenario: n_classes per task with no overlap
        random_permutation = torch.randperm(target_size)
        permutations = [random_permutation[i *
                                           data["n_classes"]: (i + 1) * data["n_classes"]] for i in range(data["n_tasks"])]

    ### NETWORK CONFIGURATION ###
    data['nn_parameters']['layers'].insert(0, torch.prod(torch.tensor(shape)))
    data['nn_parameters']['layers'].append(target_size)
    model = data["nn_type"](**data["nn_parameters"])
    net_trainer = trainer.GPUTrainer(
        batch_size=batch_size, model=model, **data, device=DEVICE)

    ### TASK SELECTION ###
    # Setting the task iterator: The idea is that we yield datasets corresponding to the framework we want to use
    # For example, if we want to use the permuted framework, we will yield datasets with permuted images, not dependant on the dataset
    for i in range(data["n_tasks"]):
        epochs = data["training_parameters"]["n_epochs"][i + k * data["n_tasks"]] if isinstance(
            data["training_parameters"]["n_epochs"], list) else data["training_parameters"]["n_epochs"]
        for epoch in range(epochs):
            num_batches = len(train_dataset) // (
                batch_size * data["n_tasks"]) if "CIL" in data["task"] or "Stream" in data["task"] else len(train_dataset) // batch_size
            train_dataset.shuffle()
            for n_batch in range(num_batches):
                batch, labels = special_task_selector(
                    data,
                    train_dataset,
                    batch_size=batch_size,
                    task_id=i,
                    iteration=n_batch,
                    max_iterations=epochs*num_batches,
                    permutations=permutations,
                    epoch=epoch,
                    continual=True
                )
                net_trainer.batch_step(batch, labels)
            if "Permuted" not in data["task"] and "CIL" not in data["task"]:
                _, _, _ = iterable_evaluation_selector(
                    data=data, test_dataset=test_dataset, net_trainer=net_trainer, permutations=permutations, train_dataset=train_dataset)
                metrics = net_trainer.testing_accuracy[-1][i].item()
                trial.report(metrics, epoch + i * epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if "Permuted" in data["task"] or "CIL" in data["task"]:
            _, _, _ = iterable_evaluation_selector(
                data=data, test_dataset=test_dataset, net_trainer=net_trainer, permutations=permutations, train_dataset=train_dataset)
            metrics = objective_computation(net_trainer, i)
            print(f"Task {i} - AA: {metrics[0]} - FT: {metrics[1]}")

    # Save all parameters of the trial in a json file
    os.makedirs(f"gridsearch/{STUDY}", exist_ok=True)
    with open(os.path.join(f"gridsearch/{STUDY}", f"{trial.number}.json"), "w") as f:
        all_accuracies = {
            f"task_{i}": net_trainer.testing_accuracy[-1][i].item() for i in range(len(net_trainer.testing_accuracy[-1]))
        }
        score = {
            "average": net_trainer.mean_testing_accuracy[-1].item(),
            "params": trial.params,
            "tasks_acc": all_accuracies,
        }
        json.dump(score, f)
    return metrics


def objective_computation(net_trainer, i):
    # 1st: Average Accuracy (AA)
    average_accuracy = net_trainer.mean_testing_accuracy[-1].item()

    # 2nd: Forward Transfer (FT) (average forgetting of all tasks)
    forward_transfer = 0
    max_of_max = []
    for task in range(i+1):
        # Max accuracy on dimension 0 for the task "task"
        max_task_acc = []
        for epoch in range(len(net_trainer.testing_accuracy)):
            max_task_acc.append(
                net_trainer.testing_accuracy[epoch][task].item())
        forward_transfer += max(max_task_acc) - \
            net_trainer.testing_accuracy[-1][task].item()
        max_of_max.append(max(max_task_acc))
    forward_transfer /= i if i != 0 else 1

    # 3rd: Highest Accuracy (HA)
    highest_accuracy = max(max_of_max)
    return average_accuracy, forward_transfer, highest_accuracy


if __name__ == "__main__":
    ### OPTUNA CONFIGURATION ###
    # Create a new study that "maximize" the accuracy of all tasks
    study = optuna.create_study(
        directions=["maximize", "minimize", "maximize"] if "Permuted" in parser.parse_args().task else [
            "maximize"],
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5,
            reduction_factor=3,
        ),
        storage=f"sqlite:///{parser.parse_args().db}",
        study_name=STUDY,
        load_if_exists=True,
    )

    study.optimize(train_iteration, n_trials=N_TRIALS)
    # Save the best trial in a json file
    trial = study.best_trial
    with open(os.path.join(f"gridsearch/{STUDY}", "best_trial.json"), "w") as f:
        output = {"number": trial.number,
                  "value": trial.value,
                  "params": trial.params,
                  "accuracies": trial.user_attrs}
        json.dump(output, f)

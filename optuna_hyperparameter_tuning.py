from utils import *
from dataloader import *
from optimizers import *
import trainer
import models
import tqdm
import torch
import os
import optuna
from optuna.trial import TrialState
import json
from models.layers.activation import Sign
import argparse


### ARGUMENTS ###
parser = argparse.ArgumentParser(description="Gridsearch for the BiNN")
parser.add_argument(
    "--device", type=str, default="0", help="Device to use (0, 1, 2, etc.), -1 is CPU")
parser.add_argument(
    "--n_trials", type=int, default=500, help="Number of trials")
parser.add_argument(
    "--study", type=str, default="gridsearch/study-default", help="Name of the study")


### GENERAL CONFIGURATION ###
DEVICE = f"cuda:{parser.parse_args().device}" if parser.parse_args(
).device != "-1" else "cpu"  # Device to use
print("STARTING GRIDSEARCH ON {}".format(DEVICE))
N_TRIALS = parser.parse_args().n_trials  # Number of trials
PADDING = 0

### PATHS ###
DATASETS_PATH = "datasets"
STUDY = parser.parse_args().study  # Name of the study


def train_iteration(trial):
    """ Train a single network

    Args:
        trial (optuna.Trial): Optuna trial
    """
    ### PARAMETERS ###
    lr = trial.suggest_float("lr", 1e-3, 10, log=True)
    scale = trial.suggest_float("scale", 1e-2, 1, log=True)
    temperature = trial.suggest_categorical("temperature", [1])
    seed = trial.suggest_categorical("seed", [1000])
    epochs = trial.suggest_categorical("epochs", [20])
    quantization = trial.suggest_categorical("quantization", [None])
    threshold = trial.suggest_categorical("threshold", [None])
    noise = trial.suggest_categorical("noise", [0])
    batch_size = trial.suggest_categorical(
        "batch_size", [256, 512, 1024, 2048])
    task = trial.suggest_categorical("task", ["CIFAR100"])
    n_tasks = trial.suggest_categorical("n_tasks", [2])
    n_classes = trial.suggest_categorical("n_classes", [50])

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.set_default_device(DEVICE)
        torch.set_default_dtype(torch.float32)

    ### DATA TO DISPLAY IN OPTUNA ###
    data = {
        "nn_type": models.BiNN,
        "nn_parameters": {
            "layers": [2048, 2048],
            "device": DEVICE,
            "dropout": False,
            "batchnorm": True,
            "bnmomentum": 0,
            "bneps": 0,
            "init": "uniform",
            "std": 0.1,
            "bias": False,
            "latent_weights": True,
            "running_stats": False,
            "affine": False,
            "activation_function": Sign.apply,
            "output_function": "log_softmax",
        },
        "criterion": torch.nn.functional.nll_loss,
        "training_parameters": {
            'n_epochs': epochs,
            'batch_size': batch_size,
            "test_mcmc_samples": 1,
            'resize': True
        },
        "optimizer": BinaryHomosynapticUncertaintyTest,
        "optimizer_parameters": {
            "lr": lr,
            "scale": scale,
            "gamma": temperature,
            "noise": noise,
            "quantization": quantization,
            "threshold": threshold,
            "update": 1,
            "num_mcmc_samples": 1,
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
    train_loader, test_loader, shape, target_size = task_selection(loader=loader,
                                                                   task=data["task"],
                                                                   n_tasks=data["n_tasks"],
                                                                   batch_size=batch_size,
                                                                   resize=resize)

    ### NETWORK CONFIGURATION ###
    data['nn_parameters']['layers'].insert(0, torch.prod(torch.tensor(shape)))
    data['nn_parameters']['layers'].append(target_size)
    model = data["nn_type"](**data["nn_parameters"])

    ### TRAINER SELECTION ###
    if data["optimizer"] in [BinaryHomosynapticUncertaintyTest]:
        net_trainer = trainer.BayesTrainer(batch_size=batch_size,
                                           model=model, **data, device=DEVICE)
    else:
        net_trainer = trainer.GPUTrainer(batch_size=batch_size,
                                         model=model, **data, device=DEVICE)

    ### TASK SELECTION ###
    task_iterator = None
    if data["task"] == "PermutedMNIST":
        permutations = [torch.randperm(torch.prod(torch.tensor(shape)))
                        for _ in range(data["n_tasks"])]
        task_iterator = train_loader[0].permute_dataset(permutations)
    elif "CIFAR" in data["task"] and data["n_tasks"] > 1:
        task_iterator = train_loader[0].class_incremental_dataset(
            n_tasks=data["n_tasks"], n_classes=data["n_classes"])
    else:
        task_iterator = train_loader

    ### TRAINING ###
    for i, task in enumerate(task_iterator):
        pbar = tqdm.trange(data["training_parameters"]["n_epochs"])
        ### STARTING TRAINING ###
        for epoch in pbar:
            net_trainer.epoch_step(task)

            ### TEST EVALUATION ###
            if data["task"] == "PermutedMNIST":
                net_trainer.evaluate(test_loader[0].permute_dataset(
                    permutations))
            elif "CIFAR" in data["task"] and data["n_tasks"] > 1:
                net_trainer.evaluate(test_loader[0].class_incremental_dataset(
                    n_tasks=data["n_tasks"], n_classes=data["n_classes"]))
            else:
                net_trainer.evaluate(test_loader)

            ### MEAN LOSS ACCURACY FOR CONTINUAL LEARNING ###
            other_tasks_stack = torch.zeros(
                len(net_trainer.testing_accuracy[-1]) - i - 1)
            other_tasks_stack.fill_(0.1)
            ongoing_accuracy = torch.cat(
                (torch.stack(net_trainer.testing_accuracy[-1][:i+1]), other_tasks_stack))
            ongoing_accuracy = ongoing_accuracy.mean()

            trial.report(
                ongoing_accuracy.item(),
                step=i*data["training_parameters"]["n_epochs"]+epoch,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

    # save all parameters of the model in a json file
    with open(os.path.join(STUDY, f"{trial.number}.json"), "w") as f:
        all_accuracies = {
            f"task_{i}": net_trainer.testing_accuracy[-1][i].item() for i in range(len(net_trainer.testing_accuracy[-1]))
        }
        score = {
            "mean_acc": net_trainer.mean_testing_accuracy[-1].item(),
            "params": trial.params,
            "tasks_acc": all_accuracies,
        }
        json.dump(score, f)
    return net_trainer.mean_testing_accuracy[-1].item()


if __name__ == "__main__":
    os.makedirs(STUDY, exist_ok=True)
    ### OPTUNA CONFIGURATION ###
    # Create a new study that "maximize" the accuracy of all tasks
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, reduction_factor=2),
        storage=f"sqlite:///{os.path.join('gridsearch', 'gridsearch-2.sqlite3')}",
        study_name=STUDY,
        load_if_exists=True,
    )

    study.optimize(train_iteration, n_trials=N_TRIALS)
    # Save the best trial in a json file
    trial = study.best_trial
    with open(os.path.join(STUDY, "best_trial.json"), "w") as f:
        output = {"value": trial.value, "params": trial.params}
        json.dump(output, f)

from numpy.random import seed as npseed
import tqdm
from torch.optim import SGD, Adam
from torch import device, cuda, functional, stack, save, prod, set_default_device, set_default_dtype, manual_seed, randperm
import datetime
from dataloader import *
from optimizers import *
import trainer
import models
import torch
import os
import optuna
import json
import argparse
from dataloader.iterable import *

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
parser.add_argument(
    "--n_classes", type=int, default=1, help="Number of classes per task")
parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--epochs", type=int, default=20, help="Number of epochs")
parser.add_argument(
    "--label_trick", type=bool, default=False, help="Label trick to use")

### GENERAL CONFIGURATION ###
DEVICE = f"cuda:{parser.parse_args().device}" if parser.parse_args(
).device != "-1" else "cpu"  # Device to use
print("STARTING GRIDSEARCH ON {}".format(DEVICE))
N_TRIALS = parser.parse_args().n_trials  # Number of trials
### PATHS ###
DATASETS_PATH = "datasets"
STUDY = parser.parse_args().study  # Name of the study


def train_iteration(trial):
    """ Train a single network

    Args:
        trial (optuna.Trial): Optuna trial
    """
    ### OPTIM PARAMETERS ###
    sigma_init = trial.suggest_float("sigma_init", 1e-3, 1e-1, log=True)
    sigma_prior = trial.suggest_float("sigma_prior", 1e-3, 1e-1, log=True)
    N_mu = trial.suggest_int("N_mu", 100, 10_000_000)
    N_sigma = trial.suggest_int("N_sigma", 100, 10_000_000)
    lr_mu = trial.suggest_float("lr_mu", 1e-3, 100, log=True)
    lr_sigma = trial.suggest_float("lr_sigma", 1e-3, 100, log=True)

    ### TASK PARAMETERS ###
    seed = trial.suggest_categorical(
        "seed", [1000])
    epochs = trial.suggest_categorical(
        "epochs", [parser.parse_args().epochs])
    batch_size = trial.suggest_categorical(
        "batch_size", [parser.parse_args().batch_size])
    task = trial.suggest_categorical(
        "task", [parser.parse_args().task])
    n_tasks = trial.suggest_categorical(
        "n_tasks", [parser.parse_args().n_tasks])
    n_classes = trial.suggest_categorical(
        "n_classes", [parser.parse_args().n_classes])
    sigma_multiplier = trial.suggest_float(
        "sigma_multiplier", 1e-1, 1e1, log=True)
    # should be an array of integers
    layers = [int(i) for i in parser.parse_args().layers.split(
        "-")] if parser.parse_args().layers != "" else []
    ### SEED ###
    npseed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    set_default_device(DEVICE)
    set_default_dtype(float32)
    ### INIT DATALOADER ###
    loader = GPULoading(device=DEVICE, root=DATASETS_PATH)
    ### NETWORK CONFIGURATION ###
    data = {
        "image_padding": 0,
        "nn_type": models.ResNet18Hybrid,
        "nn_parameters": {
            # NETWORK ###
            "layers": layers,
            # "features": [16, 32, 64],
            # "kernel_size": [3, 3, 3],
            "padding": "same",
            "device": DEVICE,
            "dropout": False,
            "bias": True,
            "n_samples_test": 5,
            "n_samples_train": 5,
            "tau": 1,
            "std": sigma_init,
            "activation_function": "relu",
            "activation_parameters": {
                "width": 1,
                "power": 2,
            },
            "normalization": "",
            "eps": 1e-5,
            "momentum": 0.15,
            "running_stats": False,
            "affine": False,
            "frozen": False,
            "sigma_multiplier": sigma_multiplier,
            "version": 0,
        },
        "training_parameters": {
            'n_epochs': epochs,
            'batch_size': batch_size,
            'test_batch_size': batch_size,
            'feature_extraction': False,
            'data_aug_it': 1,
            'full': False,
            "continual": False,
            "task_boundaries": False,
        },
        "label_trick": False,
        "output_function": "log_softmax",
        "criterion": functional.F.nll_loss,
        "reduction": "sum",
        # "optimizer": MESU,
        # "optimizer_parameters": {
        #     "sigma_prior": sigma_prior,
        #     "mu_prior": 0,
        #     "N_mu": N_mu,
        #     "N_sigma": N_sigma,
        #     "lr_mu": lr_mu,
        #     "lr_sigma": lr_sigma,
        #     "norm_term": False,
        # },
        # "optimizer": SGD,
        # "optimizer_parameters": {
        #     "lr": lr_mu,
        # },
        "optimizer": MESUDET,
        "optimizer_parameters": {
            "mu_prior": 0,
            "sigma_prior": sigma_prior,
            "N_mu": N_mu,
            "N_sigma": N_sigma,
            "c_sigma": lr_sigma,
            "c_mu": lr_mu,
            "second_order": True,
            "clamp_sigma": [0, 0],
            "clamp_mu": [0, 0],
            "enforce_learning_sigma": False,
            "normalise_grad_sigma": 0,
            "normalise_grad_mu": 0,
        },
        "task": task,
        "n_tasks": n_tasks,
        "n_classes": n_classes,
    }
    batch_size = data['training_parameters']['batch_size']
    feature_extraction = data['training_parameters']['feature_extraction'] if "feature_extraction" in data["training_parameters"] else False
    data_aug_it = data['training_parameters']['data_aug_it'] if "data_aug_it" in data["training_parameters"] else None
    ### LOADING DATASET ###
    train_dataset, test_dataset, shape, target_size = loader.task_selection(
        task=data["task"], n_tasks=data["n_tasks"], batch_size=batch_size, feature_extraction=feature_extraction, iterations=data_aug_it, padding=data["image_padding"], run=0, full=data["training_parameters"]["full"] if "full" in data["training_parameters"] else False)
    data['nn_parameters']['layers'].append(target_size)
    data['nn_parameters']['layers'].insert(0, torch.prod(torch.tensor(shape)))
    # Instantiate the network
    model = data['nn_type'](**data['nn_parameters'])
    ### INSTANTIATE THE TRAINER ###
    if data["optimizer"] in [BayesBiNN]:
        net_trainer = trainer.BayesTrainer(batch_size=batch_size,
                                           model=model, **data, device=DEVICE)
    else:
        net_trainer = trainer.GPUTrainer(batch_size=batch_size,
                                         model=model, **data, device=DEVICE)
    if isinstance(net_trainer.optimizer, MetaplasticAdam) and net_trainer.model.affine and data["training_parameters"]["task_boundaries"]:
        batch_params = []
        for i in range(data["n_tasks"]):
            batch_params.append(net_trainer.model.save_bn_states())
    ### CREATING PERMUTATIONS ###
    permutations = None
    if "PermutedLabels" in data["task"]:
        permutations = [randperm(target_size)
                        for _ in range(data["n_tasks"])]
    elif "Permuted" in data["task"]:
        permutations = [randperm(prod(tensor(shape)))
                        for _ in range(data["n_tasks"])]
    if "CIL" in data["task"]:
        # Create the permutations for the class incremental scenario: n_classes per task with no overlap
        random_permutation = randperm(target_size)
        if isinstance(data["n_classes"], int):
            permutations = [random_permutation[i * data["n_classes"]:(i + 1) * data["n_classes"]]
                            for i in range(data["n_tasks"])]
        else:
            # n_classes in a list of number of class to take per permutation, we want them to not overlap
            permutations = [random_permutation[sum(data["n_classes"][:i]):sum(data["n_classes"][:i+1])]
                            for i in range(data["n_tasks"])]
        # Create GPUTensordataset with only the class in each permutation
        if not isinstance(train_dataset, list):
            train_dataset = [train_dataset.__getclasses__(permutation)
                             for permutation in permutations]
            test_dataset = [test_dataset.__getclasses__(permutation)
                            for permutation in permutations]
    test_dataset = test_dataset if isinstance(
        test_dataset, list) else [test_dataset]
    ### TASK SELECTION ###
    for i in range(data["n_tasks"]):
        epochs = data["training_parameters"]["n_epochs"][i] if isinstance(
            data["training_parameters"]["n_epochs"], list) else data["training_parameters"]["n_epochs"]
        pbar = range(epochs)
        for epoch in pbar:
            predictions, labels = net_trainer.epoch_step(batch_size=batch_size,
                                                         test_batch_size=data["training_parameters"]["test_batch_size"],
                                                         train_dataset=train_dataset,
                                                         test_dataset=test_dataset,
                                                         task_id=i,
                                                         permutations=permutations,
                                                         epoch=epoch,
                                                         pbar=None,
                                                         epochs=epochs,
                                                         continual=data["training_parameters"]["continual"],
                                                         batch_params=batch_params if data["optimizer"] in [
                                                             MetaplasticAdam] and net_trainer.model.affine and data["training_parameters"]["task_boundaries"] else None)

            metrics = net_trainer.mean_testing_accuracy[-1].item()
            trial.report(metrics, epoch + i * epochs)
            if trial.should_prune():
                raise optuna.TrialPruned()
        ### TASK BOUNDARIES ###
        if data["training_parameters"]["task_boundaries"] == True and isinstance(net_trainer.optimizer, BayesBiNN):
            net_trainer.optimizer.update_prior_lambda()

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


if __name__ == "__main__":
    ### OPTUNA CONFIGURATION ###
    # Create a new study that "maximize" the accuracy of all tasks
    study = optuna.create_study(
        directions=["maximize"],
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

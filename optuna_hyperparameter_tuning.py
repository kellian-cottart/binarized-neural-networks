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
    lr = trial.suggest_float("lr", 0.01, 100, log=True)
    beta = trial.suggest_float("beta", 0, 0.999, step=0.001)
    gamma = trial.suggest_float("gamma", 0, 4, step=0.001)
    regularizer = trial.suggest_float("regularizer", 0, 3, step=0.01)
    seed = trial.suggest_categorical("seed", [1000])
    epochs = trial.suggest_categorical("epochs", [20])
    num_mcmc_samples = trial.suggest_categorical("num_mcmc_samples", [1])
    init_law = trial.suggest_categorical("init_law", ["gaussian"])
    # init_param = trial.suggest_float("init_param", 0, 2, step=0.01)
    init_param = trial.suggest_categorical("init_param", [0])
    temperature = trial.suggest_categorical("temperature", [1])
    batch_size = trial.suggest_categorical(
        "batch_size", [128])
    task = trial.suggest_categorical("task", [parser.parse_args().task])
    n_tasks = trial.suggest_categorical("n_tasks", [10])
    n_classes = trial.suggest_categorical("n_classes", [1])
    n_subsets = trial.suggest_categorical("n_subsets", [1])
    layer = trial.suggest_categorical("layer", [2048])
    normalization = trial.suggest_categorical(
        "normalization", ["instancenorm"])

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.set_default_device(DEVICE)
        torch.set_default_dtype(torch.float32)

    ### DATA TO DISPLAY IN OPTUNA ###
    data = {
        "nn_type": models.DNN,
        "nn_parameters": {
            "layers": [layer, layer],
            "device": DEVICE,
            "dropout": False,
            "normalization": normalization,
            "momentum": 0,
            "eps": 0,
            "init": "uniform",
            "std": 0.01,
            "bias": False,
            "running_stats": False,
            "affine": False,
            "gnnum_groups": 1,
            "activation_function": Sign.apply,
            "output_function": "log_softmax",
        },
        "criterion": torch.nn.functional.nll_loss,
        "training_parameters": {
            'n_epochs': epochs,
            'batch_size': batch_size,
            "test_mcmc_samples": 1,
            'resize': True,
            'data_aug_it': 1,
        },
        # "optimizer": torch.optim.Adam,
        # "optimizer_parameters": {
        #     "lr": lr,
        #     "weight_decay": 0,
        # },
        "optimizer": BinaryHomosynapticUncertaintyTest,
        "optimizer_parameters": {
            "lr": lr,
            "beta": beta,
            "gamma": gamma,
            "num_mcmc_samples": num_mcmc_samples,
            "init_law": init_law,
            "init_param": init_param,
            "temperature": temperature,
            "update": 1,
            "regularizer": regularizer,
        },
        "task": task,
        "n_tasks": n_tasks,
        "n_classes": n_classes,
        "n_subsets": n_subsets,
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
                                                                   resize=resize,
                                                                   iterations=data["training_parameters"]["data_aug_it"])

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
    # Setting the task iterator: The idea is that we yield datasets corresponding to the framework we want to use
    # For example, if we want to use the permuted framework, we will yield datasets with permuted images, not dependant on the dataset
    task_iterator = None
    if "Permuted" in data["task"]:
        # Create n_tasks permutations of the dataset
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
    ### TRAINING ###
    for i, task in enumerate(task_iterator):
        ### STARTING TRAINING ###
        for epoch in range(data["training_parameters"]["n_epochs"]):
            net_trainer.epoch_step(task)
            ### TEST EVALUATION ###
            # Depending on the task, we also need to use the framework on the test set and show training or not
            if "Permuted" in data["task"]:
                net_trainer.evaluate(test_loader[0].permute_dataset(
                    permutations))
            elif "CIL" in data["task"]:
                net_trainer.evaluate(test_loader[0].class_incremental_dataset(
                    permutations=permutations))
            else:
                net_trainer.evaluate(
                    test_loader)
            ### MEAN LOSS ACCURACY FOR CONTINUAL LEARNING ###
            # The idea is to put every task that has not been trained yet to 0.01, such that the mean is not affected during the optimization process of Optuna
            # If we didn't do that, tasks where the random initialization gives bad results on non-trained tasks would be pruned
            if not "Stream" in data["task"]:
                other_tasks_stack = torch.zeros(
                    len(net_trainer.testing_accuracy[-1]) - i - 1)
                other_tasks_stack.fill_(0.01)
                ongoing_accuracy = torch.cat(
                    (torch.stack(net_trainer.testing_accuracy[-1][:i+1]), other_tasks_stack)).mean()
            else:
                ongoing_accuracy = net_trainer.testing_accuracy[-1][0]
            print(
                f"Task {i+1} - Epoch {epoch+1}/{data['training_parameters']['n_epochs']} - Mean accuracy: {ongoing_accuracy.item()*100:.2f}%")
            trial.report(
                ongoing_accuracy.item(),
                step=i*data["training_parameters"]["n_epochs"]+epoch,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

    ### EXPORT PARAMETERS ###
    # Save all parameters of the trial in a json file
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
        storage=f"sqlite:///{parser.parse_args().db}",
        study_name=STUDY,
        load_if_exists=True,
    )

    study.optimize(train_iteration, n_trials=N_TRIALS)
    # Save the best trial in a json file
    trial = study.best_trial
    with open(os.path.join(STUDY, "best_trial.json"), "w") as f:
        output = {"number": trial.number,
                  "value": trial.value,
                  "params": trial.params,
                  "accuracies": trial.user_attrs}
        json.dump(output, f)

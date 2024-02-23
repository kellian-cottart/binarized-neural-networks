from utils import *
from dataloader import *
from optimizer import *
import trainer
import models

import tqdm
import torch
import os
import optuna
from optuna.trial import TrialState
import json
from models.layers.activation import Sign

### GENERAL CONFIGURATION ###
SEED = 1000  # Random seed
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PADDING = 2  # from 28x28 to 32x32
INPUT_SIZE = (28+PADDING*2)**2
N_TRIALS = 750  # Number of trials

### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"
STUDY = "gridsearch/asymmetrictanh-PermutedMNIST-Sign-Hardtanh"
ALL_GPU = True


def train_iteration(trial):
    """ Train a single network

    Args:
        config (dict): Configuration of the network
    """

    ### NETWORK CONFIGURATION ###
    model = models.BiNN(
        layers=[INPUT_SIZE, 2048, 2048, 10],  # Size of the layers
        device=DEVICE,  # Device to use
        init="uniform",  # Initialization of the weights
        std=0,  # Standard deviation of the weights
        batchnorm=True,  # Batch normalization
        bnmomentum=0,  # Batch normalization momentum
        bneps=0,  # Batch normalization epsilon
        running_stats=False,  # Batch normalization running stats
        affine=False,  # Affine
        bias=False,  # Bias
        latent_weights=False,  # Latent weights
        activation_function=Sign.apply,  # Activation function
        output_function="log_softmax",  # Output function
    )

    ### PARAMETERS ###
    lr = trial.suggest_float("lr", 1e-4, 1, log=True)
    scale = trial.suggest_float("scale", 1e-5, 1, log=True)
    temperature = trial.suggest_categorical("temperature", [1])
    seed = trial.suggest_categorical("seed", [1000])
    epochs = trial.suggest_categorical("epochs", [20])
    task = trial.suggest_categorical("task", ["PermutedMNIST"])
    quantization = trial.suggest_categorical("quantization", [None])
    threshold = trial.suggest_categorical("threshold", [None])
    noise = trial.suggest_categorical("noise", [0])

    torch.manual_seed(seed)
    if torch.cuda.is_available() and ALL_GPU:
        torch.cuda.manual_seed(seed)

    config = {
        # Fixed parameters
        "batch_size": 128,
        "epochs": epochs,
        "task": task,
        "n_tasks": 10,
        "optimizer": "BinaryHomosynapticUncertaintyTest",
    }

    if config["optimizer"] == "BinarySynapticUncertainty":
        optimizer = BinarySynapticUncertainty_OLD
    elif config["optimizer"] == "BinarySynapticUncertaintyTaskBoundaries":
        optimizer = BinarySynapticUncertaintyTaskBoundaries
    elif config["optimizer"] == "BayesBiNN":
        optimizer = BayesBiNN
    elif config["optimizer"] == "BinaryHomosynapticUncertainty":
        optimizer = BinaryHomosynapticUncertainty
    elif config["optimizer"] == "BinaryHomosynapticUncertaintyTest":
        optimizer = BinaryHomosynapticUncertaintyTest
    else:
        raise ValueError(
            f"Optimizer {config['optimizer']} not recognized")

    ### TRAINER ###
    # Creates a trainer instance that will train the model
    bayes_trainer = trainer.BayesTrainer(
        model=model,
        optimizer=optimizer,
        optimizer_parameters={
            "lr": lr,
            "scale": scale,
            "temperature": temperature,
            "gamma": 0,
            "num_mcmc_samples": 1,
            "init_lambda": 0,
            "noise": noise,
            "quantization": quantization,
            "threshold": threshold,
        },
        criterion=torch.functional.F.nll_loss,
        device=DEVICE,
        training_parameters={
            'n_epochs': config["epochs"],
            'batch_size': config["batch_size"],
            'test_mcmc_samples': 1,
        },
    )

    ### LOADER ###
    # Creates a GPULoading instance that loads any dataset in the same format
    loader = GPULoading(padding=PADDING,
                        device=DEVICE,
                        as_dataset=False)

    ### DATA ###
    mnist_train, mnist_test = mnist(loader, config["batch_size"])

    if config["task"] == "Sequential":
        fashion_train, fashion_test = fashion_mnist(
            loader, config["batch_size"])
        data_loader = [mnist_train, fashion_train]
        test_loader = [mnist_test, fashion_test]
    elif config["task"] == "PermutedMNIST":
        permutations = [torch.randperm(INPUT_SIZE)
                        for _ in range(config["n_tasks"])]
        data_loader = bayes_trainer.yield_permutation(
            mnist_train, permutations)
        test_loader = [mnist_test]

    ### TRAINING ###
    for i, task in enumerate(data_loader):
        pbar = tqdm.trange(config["epochs"])
        for epoch in pbar:
            bayes_trainer.epoch_step(task)  # Epoch of optimization
            # If permutedMNIST, permute the datasets and test
            if config["task"] == "PermutedMNIST":
                bayes_trainer.evaluate(bayes_trainer.yield_permutation(
                    test_loader[0], permutations))
            else:
                bayes_trainer.evaluate(test_loader)
            # update the progress bar
            # bayes_trainer.pbar_update(
            #     pbar, epoch, config["epochs"])
            # 1. Create a mean loss for task accuracies
            other_tasks_stack = torch.zeros(
                len(bayes_trainer.testing_accuracy[-1]) - i - 1)
            # set to 0.1
            other_tasks_stack.fill_(0.1)
            ongoing_accuracy = torch.cat(
                (torch.stack(bayes_trainer.testing_accuracy[-1][:i+1]), other_tasks_stack))
            ongoing_accuracy = ongoing_accuracy.mean()

            # 2. Maybe a second loss that evaluates the plasticity of the network

            trial.report(
                ongoing_accuracy.item(),
                step=i*config["epochs"]+epoch,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()
        if config["optimizer"] == "BinarySynapticUncertaintyTaskBoundaries" or config["optimizer"] == "BayesBiNN":
            bayes_trainer.optimizer.update_prior_lambda()

    # save all parameters of the model in a json file
    with open(os.path.join(STUDY, f"{trial.number}.json"), "w") as f:
        all_accuracies = {
            f"task_{i}": bayes_trainer.testing_accuracy[-1][i].item() for i in range(len(bayes_trainer.testing_accuracy[-1]))
        }
        score = {
            "mean_acc": bayes_trainer.mean_testing_accuracy[-1].item(),
            "params": trial.params,
            "tasks_acc": all_accuracies,
        }
        json.dump(score, f)
    return bayes_trainer.mean_testing_accuracy[-1].item()


if __name__ == "__main__":
    ### SEED ###
    os.makedirs(STUDY, exist_ok=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available() and ALL_GPU:
        torch.cuda.manual_seed(SEED)
        torch.set_default_device(DEVICE)
        torch.set_default_dtype(torch.float32)

    ### OPTUNA CONFIGURATION ###
    # Create a new study that "maximize" the accuracy of all tasks
    study = optuna.create_study(
        direction="maximize",
        # prune at quartile
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, reduction_factor=2
        ),
        storage=f"sqlite:///{os.path.join('gridsearch', 'gridsearch-2.sqlite3')}",
        study_name=STUDY,
        load_if_exists=True,
    )
    # Optimize the network for 5 trials
    study.optimize(train_iteration, n_trials=N_TRIALS)
    # Get the pruned trials and the complete trials
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    # Print the statistics of the study
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    # Save the best trial in a json file
    with open(os.path.join(STUDY, "best_trial.json"), "w") as f:
        output = {"value": trial.value, "params": trial.params}
        json.dump(output, f)

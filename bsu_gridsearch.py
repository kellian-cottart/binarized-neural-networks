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

### GENERAL CONFIGURATION ###
SEED = 2506  # Random seed
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PADDING = 2  # from 28x28 to 32x32
INPUT_SIZE = (28+PADDING*2)**2

### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"
GRIDSEARCH_PATH = "./gridsearch"
ALL_GPU = True


def train_iteration(trial):
    """ Train a single network

    Args:
        config (dict): Configuration of the network
    """
    ### PARAMETERS ###
    metaplasticity = trial.suggest_loguniform("metaplasticity", 0.1, 100)
    lr = trial.suggest_loguniform("lr", 0.1, 100)
    gamma = trial.suggest_categorical("gamma", [0])

    config = {
        # Fixed parameters
        "batch_size": 128,
        "epochs": 20,
        "task": "Sequential",
        "n_tasks": 1,
    }

    ### NETWORK CONFIGURATION ###
    model = models.BiNN(
        layers=[INPUT_SIZE, 2048, 2048, 10],  # Size of the layers
        device=DEVICE,  # Device to use
        init="uniform",  # Initialization of the weights
        std=0.1,  # Standard deviation of the weights
        batchnorm=True,  # Batch normalization
        bnmomentum=0.1,  # Batch normalization momentum
        bneps=1e-05,  # Batch normalization epsilon
        running_stats=False,  # Batch normalization running stats
        bias=False,  # Bias
        latent_weights=False,  # Latent weights
        activation_function=None,  # Activation function => None gives sign activation
        output_function="log_softmax",  # Output function
    )

    ### LOADER ###
    # Creates a GPULoading instance that loads any dataset in the same format
    loader = GPULoading(padding=PADDING,
                        device=DEVICE,
                        as_dataset=False)

    ### TRAINER ###
    # Creates a trainer instance that will train the model
    bayes_trainer = trainer.BayesTrainer(
        model=model,
        optimizer=BinarySynapticUncertaintyTaskBoundaries,
        optimizer_parameters={
            "metaplasticity": metaplasticity,
            "lr": lr,
            "temperature": 1,
            "num_mcmc_samples": 1,
            "gamma": gamma,
        },
        criterion=torch.functional.F.nll_loss,
        device=DEVICE,
        training_parameters={
            'n_epochs': config["epochs"],
            'batch_size': config["batch_size"],
            'test_mcmc_samples': 1,
        },
    )

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
    for task in data_loader:
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
            bayes_trainer.pbar_update(
                pbar, epoch, config["epochs"])
            # report the accuracy to optuna
            trial.report(
                bayes_trainer.mean_testing_accuracy[-1], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        bayes_trainer.optimizer.update_prior_lambda()

    # save trial parameters in a json file
    with open(os.path.join(GRIDSEARCH_PATH, f"trial_{trial.number}.json"), "w") as f:
        json.dump(trial.params, f)

    return bayes_trainer.mean_testing_accuracy[-1]


if __name__ == "__main__":
    ### SEED ###

    os.makedirs(GRIDSEARCH_PATH, exist_ok=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available() and ALL_GPU:
        torch.cuda.manual_seed(SEED)
        torch.set_default_device(DEVICE)
        torch.set_default_dtype(torch.float32)

    ### OPTUNA CONFIGURATION ###
    # Create a new study that "maximize" the accuracy of all tasks
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5),
    )
    # Optimize the network for 5 trials
    study.optimize(train_iteration, n_trials=5)
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
    with open(os.path.join(GRIDSEARCH_PATH, "best_trial.json"), "w") as f:
        output = {"value": trial.value, "params": trial.params}
        json.dump(output, f)

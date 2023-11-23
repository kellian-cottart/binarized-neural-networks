from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os
import wandb


### GLOBAL VARIABLES ###
BATCH_SIZE = 100  # Batch size
N_EPOCHS = 50  # Number of epochs to train on each task
LEARNING_RATE = 5e-3  # Learning rate
MIN_LEARNING_RATE = 1e-16
NAME = "Djohan - Test"


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of workers for data loading
N_TASKS = 0  # Number of tasks to train on (permutations of MNIST)
SEED = 1  # Random seed
STD = 0.1  # Standard deviation for the initialization of the weights


### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)

    ### LOAD DATASETS ###
    mnist_train, mnist_test = mnist(DATASETS_PATH, BATCH_SIZE, NUM_WORKERS)
    fashion_mnist_train, fashion_mnist_test = fashion_mnist(
        DATASETS_PATH, BATCH_SIZE, NUM_WORKERS)

    input_size = mnist_train.dataset.data.shape[1] * \
        mnist_train.dataset.data.shape[2]

    ### PIPELINE ###
    training_pipeline = []
    testing_pipeline = []

    if N_TASKS > 1:
        for i in range(N_TASKS):
            permuted_mnist_train, permuted_mnist_test = permuted_mnist(
                DATASETS_PATH, BATCH_SIZE)
            training_pipeline.append(permuted_mnist_train)
            testing_pipeline.append(permuted_mnist_test)

    else:
        training_pipeline = [mnist_train, fashion_mnist_train]
        testing_pipeline = [mnist_test, fashion_mnist_test]

        N_TASKS = len(training_pipeline)

    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "name": "BayesianNN - 100 - 100",
            "model": models.BayesianNN(
                layers=[input_size, 100, 100, 10],
                init="uniform",
                device=DEVICE,
                dropout=False),
            "training_parameters": {
                'n_epochs': N_EPOCHS
            },
            "criterion": torch.nn.NLLLoss,
            "optimizer": MESU,
            "optimizer_parameters": {
                "coeff_likeli_mu": 1,
                "coeff_likeli_sigma": 1,
                "sigma_p": 4e-2,
                "sigma_b": 15,
                "update": 3,
                "keep_prior": True
            },
            # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
            # "scheduler_parameters": {
            #     "T_max": N_EPOCHS * N_TASKS,
            #     "eta_min": MIN_LEARNING_RATE
            # },
        },
    ]

    for index, data in enumerate(networks_data):
        ### W&B INITIALIZATION ###
        ident = NAME + f" - {data['name']} - {index}"
        wandb.init(project="binarized-neural-networks", entity="kellian-cottart",
                   config=networks_data[index], name=NAME)

        ### INSTANTIATE THE TRAINER ###
        if data["optimizer"] == BayesBiNN:
            network = trainer.BayesTrainer(
                **data, device=DEVICE,)
        else:
            network = trainer.Trainer(**data, device=DEVICE,)

        ### TRAINING ###
        print(f"Training {data['name']}...")
        print(network.model)
        for train_dataset in training_pipeline:
            network.fit(
                train_dataset, **data['training_parameters'], test_loader=testing_pipeline, verbose=True)

        ### SAVING DATA ###
        full_name = os.path.join(SAVE_FOLDER, data['name'])
        folder = versionning(full_name, data['name'])
        os.makedirs(folder, exist_ok=True)

        print(f"Saving {data['name']} weights, accuracy and figure...")
        weights_name = data['name'] + "-weights"
        network.save(versionning(folder, weights_name, ".pt"))

        print(f"Saving {data['name']} accuracy...")
        accuracy_name = data['name'] + "-accuracy"
        accuracy = network.testing_accuracy
        torch.save(accuracy, versionning(
            folder, accuracy_name, ".pt"))

        print(f"Exporting visualisation of {data['name']} accuracy...")
        title = data['name'] + "-tasks-accuracy"
        visualize_sequential(title, accuracy, folder=folder)
    wandb.finish()

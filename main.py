from utils import *
import models
import torch
import trainer
from optimizer import *
import os

### GLOBAL VARIABLES ###
SEED = 2506  # Random seed
BATCH_SIZE = 100  # Batch size
STD = 0.1  # Standard deviation for the initialization of the weights
N_TASKS = 3  # Number of tasks to train on when comparing with EWC
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"


if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)

    ### LOAD DATASETS ###
    mnist_train, mnist_test = mnist(DATASETS_PATH, BATCH_SIZE)
    fashion_mnist_train, fashion_mnist_test = fashion_mnist(
        DATASETS_PATH, BATCH_SIZE)

    input_size = mnist_train.dataset.data.shape[1] * \
        mnist_train.dataset.data.shape[2]

    ### NETWORKS ###
    networks_data = {
        "BNN BiNN": {
            "model": models.BNN(
                [input_size, 2048, 2048, 2048, 2048, 10],
                init='uniform',
                std=STD,
                device=DEVICE,
                dropout=False),
            "optimizer": BayesBiNN,
            "criterion": torch.nn.CrossEntropyLoss(),
            "optimizer_parameters": {
                "lr": 1e-4,
                "beta": 0.15,
                "num_mcmc_samples": 1,
                "temperature": 1e-12,
                "scale": 5
            },
            "parameters": {'n_epochs': 10},
        }
    }

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

    for name, data in networks_data.items():

        ### INSTANTIATE THE TRAINER ###
        if data["optimizer"] == BayesBiNN:
            network = trainer.BayesTrainer(**data, device=DEVICE)
        else:
            network = trainer.Trainer(**data, device=DEVICE)

        ### TRAINING ###
        print(f"Training {name}...")
        print(network.model)
        for train_dataset in training_pipeline:
            network.fit(
                train_dataset, **data['parameters'], test_loader=testing_pipeline, verbose=True)

        ### SAVING DATA ###
        full_name = os.path.join(SAVE_FOLDER, name)
        folder = versionning(full_name, name)
        os.makedirs(folder, exist_ok=True)

        print(f"Saving {name} weights, accuracy and figure...")
        weights_name = name + "-weights"
        network.save(versionning(folder, weights_name, ".pt"))

        print(f"Saving {name} accuracy...")
        accuracy_name = name + "-accuracy"
        accuracy = network.testing_accuracy
        torch.save(accuracy, versionning(
            folder, accuracy_name, ".pt"))

        print(f"Exporting visualisation of {name} accuracy...")
        title = name + "-MNIST-FashionMNIST-accuracy"
        visualize_sequential(title, accuracy, folder=folder)

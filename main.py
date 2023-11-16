from utils import *
import models
import torch
import trainer
from optimizer import *
import os

### GLOBAL VARIABLES ###
SEED = 2506
BATCH_SIZE = 100
STD = 0.1
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
                dropout=True),
            "optimizer": BayesBiNN,
            "criterion": torch.nn.CrossEntropyLoss(),
            "optimizer_parameters": {
                "lr": 1e-4,
                "beta": 0.2,
                "num_mcmc_samples": 1,
                "temperature": 1e-12,
            },
            "parameters": {'n_epochs': 50},
        }
    }

    training_pipeline = [mnist_train, fashion_mnist_train]
    testing_pipeline = [mnist_test, fashion_mnist_test]

    for name, data in networks_data.items():

        if data["optimizer"] == BayesBiNN:
            network = trainer.BayesTrainer(**data, device=DEVICE)
        else:
            network = trainer.Trainer(**data, device=DEVICE)

        print(f"Training {name}...")
        for train_dataset in training_pipeline:
            network.fit(
                train_dataset, **data['parameters'], test_loader=testing_pipeline, verbose=True)

        # Creating folders
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

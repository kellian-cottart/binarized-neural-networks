from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt

SEED = 2506  # Random seed
N_NETWORKS = 5  # Number of networks to train
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of workers for data loading when using CPU

STD = 0.1  # Standard deviation for the initialization of the weights
PADDING = 2  # from 28x28 to 32x32
INPUT_SIZE = (28+PADDING*2)**2

### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)
    ### NETWORK CONFIGURATION ###
    networks_data = [
        # {
        #     "name": "BinaryNN-512-512"+TASK,
        #     "nn_type": models.BNN,
        #     "nn_parameters": {
        #         "layers": [input_size, 512, 512, 10],
        #         "init": "uniform",
        #         "device": DEVICE,
        #         "std": STD,
        #         "dropout": False,
        #         "batchnorm": True,
        #     },
        #     "training_parameters": {
        #         'n_epochs': N_EPOCHS
        #     },
        #     "criterion": torch.functional.F.nll_loss,
        #     "reduction": "sum",
        #     "optimizer": BayesBiNN,
        #     "optimizer_parameters": {
        #         "lr": LEARNING_RATE,
        #         "beta": 0.12,
        #         "temperature": 1e-6,
        #         "scale": 5,
        #     },
        # },
        {
            "nn_type": models.BayesianNN,
            "nn_parameters": {
                "layers": [INPUT_SIZE, 200, 200, 10],
                "device": DEVICE,
                "dropout": False,
                "batchnorm": False,
                "bias": True,
                "sigma_init": 4e-2,
                "n_samples": 10,
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 128,

            },
            "criterion": torch.nn.functional.nll_loss,
            "reduction": 'sum',
            "optimizer": MESU,
            "optimizer_parameters": {
                "coeff_likeli_mu": 1,
                "coeff_likeli_sigma": 1,
                "sigma_p": 4e-2,
                "sigma_b": 15,
                "update": 3,
            },
            # Task to train on (Sequential or PermutedMNIST)
            "task": "PermutedMNIST",
            # Number of tasks to train on (permutations of MNIST)
            "n_tasks": 10,
            "name": "BayesianNN-200-200-PermutedMNIST-10",
        },
        {
            "nn_type": models.BayesianNN,
            "nn_parameters": {
                "layers": [INPUT_SIZE, 200, 200, 10],
                "device": DEVICE,
                "dropout": False,
                "batchnorm": False,
                "bias": True,
                "sigma_init": 4e-2,
                "n_samples": 10,
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 128,

            },
            "criterion": torch.nn.functional.nll_loss,
            "reduction": 'sum',
            "optimizer": MESU,
            "optimizer_parameters": {
                "coeff_likeli_mu": 1,
                "coeff_likeli_sigma": 1,
                "sigma_p": 4e-2,
                "sigma_b": 15,
                "update": 3,
            },
            # Task to train on (Sequential or PermutedMNIST)
            "task": "Sequential",
            # Number of tasks to train on (permutations of MNIST)
            "n_tasks": 10,
            "name": "BayesianNN-200-200-Sequential",
        },
    ]

    for index, data in enumerate(networks_data):

        ### FOLDER INITIALIZATION ###
        main_folder = os.path.join(SAVE_FOLDER, data['name'])

        ### ACCURACY INITIALIZATION ###
        accuracies = []
        batch_size = data['training_parameters']['batch_size']

        ### DATASET LOADING ###
        if "cpu" in DEVICE.type:
            loader = CPULoading(DATASETS_PATH, batch_size,
                                padding=PADDING, num_workers=NUM_WORKERS)
        else:
            if torch.cuda.is_available():
                torch.set_default_device(DEVICE)
            loader = GPULoading(batch_size, padding=PADDING,
                                device=DEVICE, turbo=torch.cuda.is_available())

        ### FOR EACH NETWORK IN THE DICT ###
        for iteration in range(N_NETWORKS):
            ### SEED ###
            torch.manual_seed(SEED + iteration)

            ### NETWORK INITIALIZATION ###
            model = data['nn_type'](**data['nn_parameters'])

            ### W&B INITIALIZATION ###
            ident = f"{data['name']} - {index}"
            wandb.init(project="binarized-neural-networks", entity="kellian-cottart",
                       config=networks_data[index], name=ident)

            ### INSTANTIATE THE TRAINER ###
            if data["optimizer"] in [BinarySynapticUncertainty, BayesBiNN]:
                network = trainer.BayesTrainer(batch_size=batch_size,
                                               model=model, **data, device=DEVICE)
            else:
                network = trainer.GPUTrainer(batch_size=batch_size,
                                             model=model, **data, device=DEVICE, logarithmic=False)

            # print architecture
            print(network.model)
            ### TRAINING ###
            task = data["task"]
            n_tasks = data["n_tasks"]
            if task == "Sequential":
                mnist_train, mnist_test = mnist(loader)
                fashion_mnist_train, fashion_mnist_test = fashion_mnist(loader)
                test_loader = [mnist_test, fashion_mnist_test]
                train_loader = [mnist_train, fashion_mnist_train]
                for dataset in train_loader:
                    network.fit(
                        dataset, **data['training_parameters'], test_loader=test_loader, verbose=True)
            elif task == "PermutedMNIST":
                permutations = [torch.randperm(INPUT_SIZE)
                                for _ in range(n_tasks)]
                # Normal MNIST to permute from
                _, mnist_test = mnist(loader)
                for i in range(n_tasks):
                    # Permuted loader
                    train_dataset, _ = mnist(
                        loader, permute_idx=permutations[i])
                    network.fit(
                        train_dataset, **data['training_parameters'], test_loader=[mnist_test], verbose=True, test_permutations=permutations)

            ### SAVING DATA ###
            os.makedirs(main_folder, exist_ok=True)
            sub_folder = versionning(
                main_folder, f"{iteration}-"+data['name'], "")
            os.makedirs(sub_folder, exist_ok=True)

            print(f"Saving {data['name']} weights, accuracy and figure...")
            weights_name = data['name'] + "-weights"
            network.save(versionning(sub_folder, weights_name, ".pt"))

            print(f"Saving {data['name']} accuracy...")
            accuracy_name = data['name'] + "-accuracy"
            accuracy = network.testing_accuracy
            torch.save(accuracy, versionning(
                sub_folder, accuracy_name, ".pt"))
            accuracies.append(accuracy)

        print(f"Exporting visualisation of {data['name']} accuracy...")
        title = data['name'] + "-tasks"
        visualize_sequential(title, accuracies, folder=main_folder)
    wandb.finish()

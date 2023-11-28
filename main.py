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

### GLOBAL VARIABLES ###
BATCH_SIZE = 128  # Batch size
N_EPOCHS = 20  # Number of epochs to train on each task
LEARNING_RATE = 1e-3  # Learning rate
MIN_LEARNING_RATE = 1e-16
NAME = "BiNNBayes-metaplasticity"
N_NETWORKS = 1  # Number of networks to train
TASK = "PermutedMNIST"  # Task to train on (Sequential or PermutedMNIST)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of workers for data loading when using CPU
N_TASKS = 10  # Number of tasks to train on (permutations of MNIST)
SEED = 7  # Random seed
STD = 0.1  # Standard deviation for the initialization of the weights

# FOR NORMALIZATION
MEAN = 0
STD = 1
PADDING = 2  # from 28x28 to 32x32

### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)

    if "cpu" in DEVICE.type:
        loader = CPULoading(DATASETS_PATH, BATCH_SIZE, mean=MEAN, std=STD,
                            padding=PADDING, num_workers=NUM_WORKERS)
    else:
        if torch.cuda.is_available():
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(DEVICE)
        loader = GPULoading(BATCH_SIZE, mean=MEAN, std=STD,
                            padding=PADDING, device=DEVICE, turbo=torch.cuda.is_available())

    input_size = (28+PADDING*2)**2

    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "name": "DNN-1024-1024",
            "nn_type": models.BNN,
            "nn_parameters": {
                "layers": [input_size, 1024, 1024, 10],
                "init": "uniform",
                "device": DEVICE,
                "dropout": False,
                "batchnorm": True,
            },
            "training_parameters": {
                'n_epochs': N_EPOCHS
            },
            "criterion": torch.functional.F.nll_loss,
            "reduction": "mean",
            "optimizer": torch.optim.Adam,
            "optimizer_parameters": {
                "lr": LEARNING_RATE,
            },
        },
        # {
        #     "name": "BinaryNN-1024-1024",
        #     "nn_type": models.BNN,
        #     "nn_parameters": {
        #         "layers": [input_size, 1024, 1024, 10],
        #         "init": "uniform",
        #         "device": DEVICE,
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
        #         "beta": 0.15,
        #         "temperature": 1e-4,
        #         "num_mcmc_samples": 1,
        #     },
        # },
        # {
        #     "name": "BayesianNN-200-200-PermutedMNIST",
        #     "nn_type": models.BayesianNN,
        #     "nn_parameters": {
        #         "layers": [input_size, 200, 200, 10],
        #         "device": DEVICE,
        #         "dropout": False,
        #         "batchnorm": False,
        #         "bias": True,
        #         "sigma_init": 4e-2,
        #         "n_samples": 10,
        #     },
        #     "training_parameters": {
        #         'n_epochs': N_EPOCHS
        #     },
        #     "criterion": torch.nn.functional.nll_loss,
        #     "reduction": 'mean',
        #     "optimizer": MESU,
        #     "optimizer_parameters": {
        #         "coeff_likeli_mu": 1,
        #         "coeff_likeli_sigma": 1,
        #         "sigma_p": 4e-2,
        #         "sigma_b": 15,
        #         "update": 3,
        #     },
        # },
    ]

    for index, data in enumerate(networks_data):

        ### FOLDER INITIALIZATION ###
        main_folder = os.path.join(SAVE_FOLDER, data['name'])
        os.makedirs(main_folder, exist_ok=True)

        ### ACCURACY INITIALIZATION ###
        accuracies = []
        for iteration in range(N_NETWORKS):
            ### SEED ###
            torch.manual_seed(SEED + iteration)

            ### NETWORK INITIALIZATION ###
            model = data['nn_type'](**data['nn_parameters'])

            ### W&B INITIALIZATION ###
            ident = NAME + f" - {data['name']} - {index}"
            wandb.init(project="binarized-neural-networks", entity="kellian-cottart",
                       config=networks_data[index], name=NAME)

            ### INSTANTIATE THE TRAINER ###
            if data["optimizer"] in [BinarySynapticUncertainty, BayesBiNN]:
                network = trainer.BayesTrainer(batch_size=BATCH_SIZE,
                                               model=model, **data, device=DEVICE)
            else:
                network = trainer.GPUTrainer(batch_size=BATCH_SIZE,
                                             model=model, **data, device=DEVICE,)

            # print architecture
            print(network.model)

            ### TRAINING ###
            if TASK == "Sequential":
                mnist_train, mnist_test = mnist(loader, permute_idx=None)
                fashion_mnist_train, fashion_mnist_test = fashion_mnist(loader)
                test_loader = [mnist_test, fashion_mnist_test]
                train_loader = [mnist_train, fashion_mnist_train]
                for dataset in train_loader:
                    network.fit(
                        dataset, **data['training_parameters'], test_loader=test_loader, verbose=True)
            elif TASK == "PermutedMNIST":
                permutations = [torch.randperm(input_size)
                                for _ in range(N_TASKS)]
                for i in range(N_TASKS):
                    # N task and N+1 task to slowly shift the distribution
                    _, mnist_test = mnist(
                        loader, permute_idx=permutations[i])
                    train_dataset, _ = mnist(
                        loader, permute_idx=permutations[i])
                    network.fit(
                        train_dataset, **data['training_parameters'], test_loader=[mnist_test], verbose=True, test_permutations=permutations)

            ### SAVING DATA ###
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

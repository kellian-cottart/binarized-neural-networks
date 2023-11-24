from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os
import wandb


### GLOBAL VARIABLES ###
BATCH_SIZE = 128  # Batch size
N_EPOCHS = 100  # Number of epochs to train on each task
LEARNING_RATE = 1e-3  # Learning rate
MIN_LEARNING_RATE = 1e-16
NAME = "BiNNBayes-metaplasticity"
N_NETWORKS = 1  # Number of networks to train

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8  # Number of workers for data loading
N_TASKS = 0  # Number of tasks to train on (permutations of MNIST)
SEED = 1  # Random seed
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

    ### LOAD DATASETS ###
    if "cpu" in DEVICE.type:
        loader = CPULoading(DATASETS_PATH, BATCH_SIZE, mean=MEAN, std=STD,
                            padding=PADDING, num_workers=NUM_WORKERS)
        mnist_train, mnist_test = loader(datasets.MNIST)
        fashion_mnist_train, fashion_mnist_test = loader(
            datasets.FashionMNIST)
    else:
        gpu_loader = GPULoading(BATCH_SIZE, mean=MEAN, std=STD,
                                padding=PADDING, device=DEVICE)
        mnist_train, mnist_test = gpu_loader(
            path_train_x="datasets/MNIST/raw/train-images-idx3-ubyte",
            path_train_y="datasets/MNIST/raw/train-labels-idx1-ubyte",
            path_test_x="datasets/MNIST/raw/t10k-images-idx3-ubyte",
            path_test_y="datasets/MNIST/raw/t10k-labels-idx1-ubyte",
        )

        fashion_mnist_train, fashion_mnist_test = gpu_loader(
            path_train_x="datasets/FashionMNIST/raw/train-images-idx3-ubyte",
            path_train_y="datasets/FashionMNIST/raw/train-labels-idx1-ubyte",
            path_test_x="datasets/FashionMNIST/raw/t10k-images-idx3-ubyte",
            path_test_y="datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte",
        )

    input_size = mnist_train.dataset.data.shape[1] * \
        mnist_train.dataset.data.shape[2]

    ### PIPELINE ###
    training_pipeline = []
    testing_pipeline = []

    if N_TASKS > 1:
        for i in range(N_TASKS):
            permuted_mnist_train, permuted_mnist_test = loader(
                PermutedMNIST, permute_idx=torch.randperm(input_size))
            training_pipeline.append(permuted_mnist_train)
            testing_pipeline.append(permuted_mnist_test)

    else:
        training_pipeline = [mnist_train, fashion_mnist_train]
        testing_pipeline = [mnist_test, fashion_mnist_test]

        N_TASKS = len(training_pipeline)

    ### NETWORK CONFIGURATION ###
    networks_data = [
        # {
        #     "name": "BinaryNN-1024-1024",
        #     "nn_type": models.BNN,
        #     "nn_parameters": {
        #         "layers": [input_size, 1024, 1024, 10],
        #         "init": "uniform",
        #         "device": DEVICE,
        #         "dropout": False
        #     },
        #     "training_parameters": {
        #         'n_epochs': N_EPOCHS
        #     },
        #     "criterion": torch.nn.NLLLoss(),
        #     "optimizer": BinarySynapticUncertainty,
        #     "optimizer_parameters": {
        #         "lr": LEARNING_RATE,
        #         "beta": 0.15,
        #         "temperature": 1e-7,
        #         "num_mcmc_samples": 1,
        #     },
        #     # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        #     # "scheduler_parameters": {
        #     #     "T_max": N_EPOCHS * N_TASKS,
        #     #     "eta_min": MIN_LEARNING_RATE
        #     # },
        # },
        {
            "name": "BayesianNN-200-200",
            "nn_type": models.BayesianNN,
            "nn_parameters": {
                "layers": [input_size, 200, 200, 10],
                "device": DEVICE,
                "dropout": False,
                "sigma_init": 0.04,
                "n_samples": 5
            },
            "training_parameters": {
                'n_epochs': N_EPOCHS
            },
            "criterion": torch.nn.NLLLoss(),
            "optimizer": MESU,
            "optimizer_parameters": {
                "coeff_likeli_mu": 1,
                "coeff_likeli_sigma": 1,
                "sigma_p": 4e-2,
                "sigma_b": 10,
                "update": 3,
                "keep_prior": True,
            },
            # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
            # "scheduler_parameters": {
            #     "T_max": N_EPOCHS * N_TASKS,
            #     "eta_min": MIN_LEARNING_RATE
            # },
        },
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
                network = trainer.BayesTrainer(
                    model=model, **data, device=DEVICE,)
            else:
                network = trainer.Trainer(model=model, **data, device=DEVICE,)

            ### TRAINING ###
            print(f"Training {data['name']}...")
            print(network.model)
            for train_dataset in training_pipeline:
                network.fit(
                    train_dataset, **data['training_parameters'], test_loader=testing_pipeline, verbose=True)

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

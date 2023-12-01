from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os

SEED = 0  # Random seed
N_NETWORKS = 1  # Number of networks to train
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.set_default_device(DEVICE)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "name": "BinaryNN-1024-1024-Sequential-Metaplastic",
            "nn_type": models.BNN,
            "nn_parameters": {
                "layers": [INPUT_SIZE, 1024, 1024, 10],
                "init": "uniform",
                "device": DEVICE,
                "std": STD,
                "dropout": False,
                "batchnorm": True,
            },
            "training_parameters": {
                'n_epochs': 50,
                'batch_size': 128,
            },
            "criterion": torch.functional.F.nll_loss,
            "reduction": "mean",
            "optimizer": MetaplasticAdam,
            "optimizer_parameters": {
                "lr": 1e-3,
                "metaplasticity": 15,
                "weight_decay": 1e-8,
            },
            "task": "Sequential",
            "n_tasks": 10,
            "padding": PADDING,
        },
        # {
        #     "name": "BinaryNN-100-100-PermutedMNIST",
        #     "nn_type": models.BNN,
        #     "nn_parameters": {
        #         "layers": [INPUT_SIZE, 100, 100, 10],
        #         "init": "uniform",
        #         "device": DEVICE,
        #         "std": STD,
        #         "dropout": True,
        #         "batchnorm": True,
        #     },
        #     "training_parameters": {
        #         'n_epochs': 100,
        #         'batch_size': 128,
        #         'test_mcmc_samples': 100,
        #     },
        #     "criterion": torch.functional.F.nll_loss,
        #     "reduction": "mean",
        #     "optimizer": BayesBiNN,
        #     "optimizer_parameters": {
        #         "lr": 1e-3,
        #         "beta": 0,
        #         "temperature": 1e-2,
        #         "num_mcmc_samples": 1,
        #     },
        #     "task": "PermutedMNIST",
        #     "n_tasks": 10,
        #     "padding": PADDING,
        # },
        # {
        #     "nn_type": models.BayesianNN,
        #     "nn_parameters": {
        #         "layers": [INPUT_SIZE, 200, 200, 10],
        #         "device": DEVICE,
        #         "dropout": False,
        #         "batchnorm": False,
        #         "bias": True,
        #         "sigma_init": 4e-2,
        #         "n_samples": 10,
        #     },
        #     "training_parameters": {
        #         'n_epochs': 50,
        #         'batch_size': 128,
        #     },
        #     "criterion": torch.nn.functional.nll_loss,
        #     "reduction": 'sum',
        #     "optimizer": MESU,
        #     "optimizer_parameters": {
        #         "coeff_likeli_mu": 1,
        #         "coeff_likeli_sigma": 1,
        #         "sigma_p": 4e-2,
        #         "sigma_b": 15,
        #         "update": 3,
        #         "keep_prior": False,
        #     },
        #     # Task to train on (Sequential or PermutedMNIST)
        #     "task": "Sequential",
        #     # Number of tasks to train on (permutations of MNIST)
        #     "n_tasks": 10,
        #     "name": "BayesianNN-200-200-Sequential",
        #     "padding": PADDING,
        # },
    ]

    for index, data in enumerate(networks_data):

        ### FOLDER INITIALIZATION ###
        main_folder = os.path.join(SAVE_FOLDER, data['name'])

        ### ACCURACY INITIALIZATION ###
        accuracies = []
        batch_size = data['training_parameters']['batch_size']
        padding = data['padding'] if 'padding' in data else 0
        ### DATASET LOADING ###
        loader = GPULoading(padding=padding,
                            device=DEVICE, as_dataset=False)
        # loader = CPULoading(DATASETS_PATH, padding=padding,
        #                     num_workers=NUM_WORKERS)

        ### FOR EACH NETWORK IN THE DICT ###
        for iteration in range(N_NETWORKS):
            ### SEED ###
            torch.manual_seed(SEED + iteration)

            ### NETWORK INITIALIZATION ###
            model = data['nn_type'](**data['nn_parameters'])

            ### W&B INITIALIZATION ###
            ident = f"{data['name']} - {index}"

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
                mnist_train, mnist_test = mnist(loader, batch_size)
                fashion_mnist_train, fashion_mnist_test = fashion_mnist(
                    loader, batch_size)
                test_loader = [mnist_test, fashion_mnist_test]
                train_loader = [mnist_train, fashion_mnist_train]
                for dataset in train_loader:
                    network.fit(
                        dataset, **data['training_parameters'], test_loader=test_loader, verbose=True)
            elif task == "PermutedMNIST":
                permutations = [torch.randperm(INPUT_SIZE)
                                for _ in range(n_tasks)]
                # Normal MNIST to permute from
                _, mnist_test = mnist(loader, batch_size)
                for i in range(n_tasks):
                    # Permuted loader
                    train_dataset, _ = mnist(
                        loader, batch_size=batch_size, permute_idx=permutations[i])
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

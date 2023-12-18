from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os
import json

SEED = 2506  # Random seed
N_NETWORKS = 1  # Number of networks to train
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Number of workers for data loading when using CPU

PADDING = 0  # from 28x28 to 32x32
INPUT_SIZE = (28+PADDING*2)**2

### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"
ALL_GPU = True

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)
    if torch.cuda.is_available() and ALL_GPU:
        torch.cuda.manual_seed(SEED)
        torch.set_default_device(DEVICE)
        torch.set_default_dtype(torch.float32)

    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "nn_type": models.BiNN,
            "nn_parameters": {
                "layers": [INPUT_SIZE, 4096, 4096, 10],
                "init": "uniform",
                "device": DEVICE,
                "std": 0.1,
                "dropout": False,
                "batchnorm": True,
                "bnmomentum": 0.1,
                "bneps": 1e-05,
                "bias": False,
                "latent_weights": True,
                "running_stats": False,
                "activation_function": None,
                "output_function": "log_softmax",
            },
            "training_parameters": {
                'n_epochs': 50,
                'batch_size': 128,
                'test_mcmc_samples': 1,
            },
            "criterion": torch.functional.F.nll_loss,
            "optimizer": MetaplasticAdam,
            "optimizer_parameters": {
                "metaplasticity": 1.5,
                "lr": 0.005,
                "weight_decay": 1e-8,
            },
            # "optimizer": BinarySynapticUncertainty,
            # "optimizer_parameters": {
            #     "metaplasticity": 0.35,
            #     "lr": 1,
            #     "temperature": 1,
            #     "gamma": 0,
            #     "num_mcmc_samples": 1,
            # },
            "task": "Sequential",
            "n_tasks": 10,
            "padding": PADDING,
        },
    ]

    for index, data in enumerate(networks_data):

        ### NAME INITIALIZATION ###
        # name should be optimizer-layer2-layer3-...-layerN-1-task-metaplacity
        name = f"{data['optimizer'].__name__}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers'][1:-1]])}-{data['task']}"
        # add some parameters to the name
        for key, value in data['optimizer_parameters'].items():
            name += f"-{key}-{value}"
        print(f"Training {name}...")

        ### FOLDER INITIALIZATION ###
        main_folder = os.path.join(SAVE_FOLDER, name)

        ### ACCURACY INITIALIZATION ###
        accuracies = []
        batch_size = data['training_parameters']['batch_size']
        padding = data['padding'] if 'padding' in data else 0

        ### DATASET LOADING ###
        if ALL_GPU:
            loader = GPULoading(padding=padding,
                                device=DEVICE, as_dataset=False)
        else:
            loader = CPULoading(DATASETS_PATH, padding=padding,
                                num_workers=NUM_WORKERS)

        ### FOR EACH NETWORK IN THE DICT ###
        for iteration in range(N_NETWORKS):
            ### SEED ###
            if iteration != 0:
                torch.manual_seed(SEED + iteration)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(SEED + iteration)

            ### NETWORK INITIALIZATION ###
            model = data['nn_type'](**data['nn_parameters'])

            ### W&B INITIALIZATION ###
            ident = f"{name} - {index}"

            ### INSTANTIATE THE TRAINER ###
            if data["optimizer"] in [BinarySynapticUncertainty, BayesBiNN]:
                network = trainer.BayesTrainer(batch_size=batch_size,
                                               model=model, **data, device=DEVICE)
            else:
                network = trainer.GPUTrainer(batch_size=batch_size,
                                             model=model, **data, device=DEVICE)
            print(network.model)

            ### TRAINING ###
            task = data["task"]
            if task == "Sequential":
                mnist_train, mnist_test = mnist(loader, batch_size)
                fashion_mnist_train, fashion_mnist_test = fashion_mnist(
                    loader, batch_size)
                test_loader = [mnist_test, fashion_mnist_test]
                train_loader = [mnist_train, fashion_mnist_train]
                for i, dataset in enumerate(train_loader):
                    network.fit(
                        dataset,
                        **data['training_parameters'],
                        test_loader=test_loader,
                        verbose=True,
                        name_loader=["t1_MNIST", "t2_FashionMNIST"]
                    )
            elif task == "PermutedMNIST":
                n_tasks = data["n_tasks"]
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
                main_folder, f"{iteration}-"+name, "")
            os.makedirs(sub_folder, exist_ok=True)

            print(f"Saving {name} weights, accuracy and figure...")
            weights_name = name + "-weights"
            network.save(versionning(sub_folder, weights_name, ".pt"))

            print(f"Saving {name} configuration...")
            # dump data as json, and turn into string all non-json serializable objects
            json_data = json.dumps(data, default=lambda o: str(o))
            config_name = name + "-config"
            with open(versionning(sub_folder, config_name, ".json"), "w") as f:
                f.write(json_data)

            print(f"Saving {name} accuracy...")
            accuracy_name = name + "-accuracy"
            accuracy = network.testing_accuracy
            torch.save(accuracy, versionning(
                sub_folder, accuracy_name, ".pt"))
            accuracies.append(accuracy)

        print(f"Exporting visualisation of {name} accuracy...")
        title = name + "-tasks"
        visualize_sequential(title, accuracies, folder=main_folder, sequential=True if task ==
                             "Sequential" else False)

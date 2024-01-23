from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os
import json
import tqdm

SEED = 1000  # Random seed
N_NETWORKS = 1  # Number of networks to train
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Number of workers for data loading when using CPU

PADDING = 2  # from 28x28 to 32x32
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
                "layers": [INPUT_SIZE, 2048, 2048, 10],
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
            "optimizer": BSUTest,
            "optimizer_parameters": {
                "lr": 10,
                "gamma": 0.1,
                "temperature": 1,
                "num_mcmc_samples": 1,
                "init_lambda": 0.1,
            },
            "task": "Sequential",
            "n_tasks": 1,  # PermutedMNIST: number of tasks, Sequential: number of mnist, fashion_mnist pairs
            "padding": PADDING,
        }
    ]

    for index, data in enumerate(networks_data):

        ### NAME INITIALIZATION ###
        # name should be optimizer-layer2-layer3-...-layerN-1-task-metaplac ity
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
        loader = GPULoading(padding=padding,
                            device=DEVICE,
                            as_dataset=False)

        ### FOR EACH NETWORK IN THE DICT ###
        for iteration in range(N_NETWORKS):
            ### INIT ###
            if iteration != 0:
                torch.manual_seed(SEED + iteration)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(SEED + iteration)
            model = data['nn_type'](**data['nn_parameters'])
            ident = f"{name} - {index}"

            ### INSTANTIATE THE TRAINER ###
            if data["optimizer"] in [BinarySynapticUncertainty, BayesBiNN, BinarySynapticUncertaintyTaskBoundaries, BSUTest]:
                net_trainer = trainer.BayesTrainer(batch_size=batch_size,
                                                   model=model, **data, device=DEVICE)
            else:
                net_trainer = trainer.GPUTrainer(batch_size=batch_size,
                                                 model=model, **data, device=DEVICE)
            print(net_trainer.model)

            ### TRAINING ###
            mnist_train, mnist_test = mnist(loader, batch_size)
            if data["task"] == "Sequential":
                fashion_train, fashion_test = fashion_mnist(
                    loader, batch_size)
                train_loader = [mnist_train, fashion_train]
                test_loader = [mnist_test, fashion_test]
            elif data["task"] == "PermutedMNIST":
                permutations = [torch.randperm(INPUT_SIZE)
                                for _ in range(data["n_tasks"])]
                train_loader = net_trainer.yield_permutation(
                    mnist_train, permutations)
                test_loader = [mnist_test]
            else:
                raise ValueError(
                    f"Task {data['task']} is not implemented. Please choose between Sequential and PermutedMNIST")

            for i, task in enumerate(train_loader):
                pbar = tqdm.trange(data["training_parameters"]["n_epochs"])
                for epoch in pbar:
                    # If BSUTest, visualize lambda
                    if data["optimizer"] == BSUTest and epoch % 10 == 0:
                        net_trainer.optimizer.visualize_lambda(
                            path=os.path.join(main_folder, "lambda"),
                            threshold=100,
                        )
                    net_trainer.epoch_step(task)  # Epoch of optimization
                    # If permutedMNIST, permute the datasets and test
                    if data["task"] == "PermutedMNIST":
                        net_trainer.evaluate(net_trainer.yield_permutation(
                            test_loader[0], permutations))
                    else:
                        net_trainer.evaluate(test_loader)
                    # update the progress bar
                    net_trainer.pbar_update(
                        pbar, epoch, data["training_parameters"]["n_epochs"])
                ### TASK BOUNDARIES ###
                if data["optimizer"] in [BinarySynapticUncertaintyTaskBoundaries, BayesBiNN]:
                    net_trainer.optimizer.update_prior_lambda()
                if data["optimizer"] == MetaplasticAdam:
                    net_trainer.reset_optimizer(data['optimizer_parameters'])

            ### SAVING DATA ###
            os.makedirs(main_folder, exist_ok=True)
            sub_folder = versionning(
                main_folder, f"{iteration}-"+name, "")
            os.makedirs(sub_folder, exist_ok=True)

            print(f"Saving {name} weights, accuracy and figure...")
            weights_name = name + "-weights"
            net_trainer.save(versionning(sub_folder, weights_name, ".pt"))

            print(f"Saving {name} configuration...")
            # dump data as json, and turn into string all non-json serializable objects
            json_data = json.dumps(data, default=lambda o: str(o))
            config_name = name + "-config"
            with open(versionning(sub_folder, config_name, ".json"), "w") as f:
                f.write(json_data)

            print(f"Saving {name} accuracy...")
            accuracy_name = name + "-accuracy"
            accuracy = net_trainer.testing_accuracy
            torch.save(accuracy, versionning(
                sub_folder, accuracy_name, ".pt"))
            accuracies.append(accuracy)

        print(f"Exporting visualisation of {name} accuracy...")
        title = name + "-tasks"
        visualize_sequential(title, accuracies, folder=main_folder, sequential=True if data['task'] ==
                             "Sequential" else False)

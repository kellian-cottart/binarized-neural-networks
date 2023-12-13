from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os
import json
from sklearn.datasets import make_moons

SEED = 2506  # Random seed
N_NETWORKS = 1  # Number of networks to train
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### PATHS ###
SAVE_FOLDER = "saved-two-moons"
DATASETS_PATH = "datasets"

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.set_default_device(DEVICE)
        torch.set_default_dtype(torch.float32)

    ### CREATE TWO MOONS DATASET ###
    X, y = make_moons(n_samples=1152, noise=0.1, random_state=SEED)
    X_train = X[:1024]
    y_train = y[:1024]
    X_test = X[1024:]
    y_test = y[1024:]

    train_tensor = GPUTensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_tensor = GPUTensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = GPUDataLoader(train_tensor, batch_size=128, shuffle=True)
    test_loader = GPUDataLoader(test_tensor, batch_size=1152-1024)
    INPUT_SIZE = train_tensor.data.shape[1]

    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "nn_type": models.BNN,
            "nn_parameters": {
                "layers": [INPUT_SIZE, 2048, 2048, 10],
                "init": "uniform",
                "device": DEVICE,
                "std": 0.1,
                "dropout": False,
                "batchnorm": True,
                "bnmomentum": 0.15,
                "bneps": 1e-5,
                "latent_weights": False,
                "running_stats": False,
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 128,
                'test_mcmc_samples': 1,
            },
            "criterion": torch.functional.F.nll_loss,
            "reduction": "mean",
            "optimizer": BinarySynapticUncertainty,
            "optimizer_parameters": {
                "temperature": 1,
                "num_mcmc_samples": 1,
                "init_lambda": 0,
                "lr": lr,
                "metaplasticity": metaplasticity,
                "gamma": 0,
            },
        } for metaplasticity in torch.linspace(0.1, 5, 10) for lr in [0.1, 0.01, 0.001]
    ]

    for index, data in enumerate(networks_data):

        # name should be optimizer-layer2-layer3-...-layerN-1-task-metaplacity
        name = f"{data['optimizer'].__name__}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers'][1:-1]])}"
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
        ### FOR EACH NETWORK IN THE DICT ###
        for iteration in range(N_NETWORKS):
            ### SEED ###
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
                                             model=model, **data, device=DEVICE, logarithmic=True)
            print(network.model)

            ### TRAINING ###

            network.fit(
                train_loader, **data['training_parameters'], test_loader=test_loader, verbose=True)

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

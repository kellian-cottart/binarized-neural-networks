from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizer import *
import os
import json
import tqdm
from models.layers.activation import Sign

SEED = 1000  # Random seed
N_NETWORKS = 1  # Number of networks to train
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Number of workers for data loading when using CPU
PADDING = 0


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

    ### INIT DATALOADER ###
    loader = GPULoading(padding=PADDING,
                        device=DEVICE,
                        as_dataset=False)

    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "nn_type": models.BiNN,
            "nn_parameters": {
                "layers": [2048, 2048, 10],
                "device": DEVICE,
                "dropout": False,
                "batchnorm": True,
                "bnmomentum": 0,
                "bneps": 0,
                "init": "uniform",
                "std": 0,
                "bias": False,
                "latent_weights": False,
                "running_stats": False,
                "affine": False,
                "activation_function": Sign.apply,
                "output_function": "log_softmax",
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 128,
                "test_mcmc_samples": 1,
            },
            "criterion": torch.functional.F.nll_loss,
            "optimizer": BinaryHomosynapticUncertaintyTest,
            "optimizer_parameters": {
                "lr": 0.0374,
                "scale": 0.06,
                "gamma": 0,
                "noise": 0,
                "quantization": None,
                "threshold": None,
            },
            "task": "CIFAR10",
            "n_tasks": 1,
            # PermutedMNIST: number of tasks,
            # Sequential: number of mnist, fashion_mnist pairs
            # CIFAR10: 1
        }
    ]
    for index, data in enumerate(networks_data):

        ### NAME INITIALIZATION ###
        # name should be optimizer-layer2-layer3-...-layerN-1-task-metaplac ity
        name = f"{data['optimizer'].__name__}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers'][1:-1]])}-{data['task']}-{data['nn_parameters']['activation_function'].__name__}"

        # add some parameters to the name
        for key, value in data['optimizer_parameters'].items():
            name += f"-{key}-{value}"
        print(f"Training {name}...")

        ### FOLDER INITIALIZATION ###
        main_folder = os.path.join(SAVE_FOLDER, name)

        ### ACCURACY INITIALIZATION ###
        accuracies = []
        batch_size = data['training_parameters']['batch_size']

        ### FOR EACH NETWORK IN THE DICT ###
        for iteration in range(N_NETWORKS):

            ### INIT DATASET ###
            if data["task"] == "Sequential":
                mnist_train, mnist_test = mnist(loader, batch_size)
                input_size = mnist_train.dataset[0][0].shape[0]

                fashion_train, fashion_test = fashion_mnist(
                    loader, batch_size)
                train_loader = [mnist_train, fashion_train]
                test_loader = [mnist_test, fashion_test]
            elif data["task"] == "PermutedMNIST":
                mnist_train, mnist_test = mnist(loader, batch_size)
                input_size = mnist_train.dataset[0][0].shape[0]
                permutations = [torch.randperm(input_size)
                                for _ in range(data["n_tasks"])]
                train_loader = net_trainer.yield_permutation(
                    mnist_train, permutations)
                test_loader = [mnist_test]
            elif data["task"] == "CIFAR10":
                cifar10_train, cifar10_test = cifar10(
                    loader, batch_size=batch_size)
                input_size = cifar10_train.dataset[0][0].shape[0]
                train_loader = [cifar10_train]
                test_loader = [cifar10_test]
            else:
                raise ValueError(
                    f"Task {data['task']} is not implemented.")

            ### INIT NETWORK ###
            if iteration != 0:
                torch.manual_seed(SEED + iteration)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(SEED + iteration)
            # add input size to the layer of the network parameters
            data['nn_parameters']['layers'].insert(0, input_size)
            # instantiate the network
            model = data['nn_type'](**data['nn_parameters'])
            ident = f"{name} - {index}"

            ### INSTANTIATE THE TRAINER ###
            if data["optimizer"] in [BinarySynapticUncertainty_OLD, BayesBiNN, BinarySynapticUncertaintyTaskBoundaries, BinaryHomosynapticUncertainty, BinaryHomosynapticUncertaintyTest]:
                net_trainer = trainer.BayesTrainer(batch_size=batch_size,
                                                   model=model, **data, device=DEVICE)
            else:
                net_trainer = trainer.GPUTrainer(batch_size=batch_size,
                                                 model=model, **data, device=DEVICE)
            print(net_trainer.model)

            for i, task in enumerate(train_loader):
                pbar = tqdm.trange(data["training_parameters"]["n_epochs"])
                for epoch in pbar:
                    # If BinaryHomosynapticUncertainty, visualize lambda
                    if data["optimizer"] in [BinaryHomosynapticUncertainty, BinaryHomosynapticUncertaintyTest] and epoch % 10 == 0:
                        net_trainer.optimizer.visualize_lambda(
                            path=os.path.join(main_folder, "lambda"),
                            threshold=25,
                        )
                    if data["optimizer"] == BinaryMetaplasticUncertainty and epoch % 10 == 0:
                        for param in model.parameters():
                            visualize_lambda(
                                lambda_=param,
                                path=os.path.join(main_folder, "lambda"),
                                threshold=25,
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
        # if number of tasks is 100, export the accuracy of the first 10 and last 10 tasks
        if data['n_tasks'] >= 10:
            title = name + "-tasks-1-10"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=1, t_end=10)
        if data['n_tasks'] == 100:
            title = name + "-tasks-91-100"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=91, t_end=100)

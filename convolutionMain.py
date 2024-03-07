from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizers import *
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
SAVE_FOLDER = "saved_convolutional_models"
DATASETS_PATH = "datasets"
ALL_GPU = True

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)
    if torch.cuda.is_available() and ALL_GPU:
        torch.cuda.manual_seed(SEED)
        torch.set_default_device(DEVICE)

    ### INIT DATALOADER ###
    loader = GPULoading(padding=PADDING,
                        device=DEVICE,
                        as_dataset=False)

    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "nn_type": models.ConvBiNN,
            "nn_parameters": {
                "layers": [2048, 2048],
                "features": [32, 64, 128],
                "kernel_size": (3, 3),
                "padding": "same",
                "stride": 1,
                "activation_function": Sign.apply,
                "output_function": "log_softmax",
                "dropout": False,
                "batchnorm": True,
                "bnmomentum": 0,
                "bneps": 0,
                "init": "uniform",
                "std": 0.1,
                "bias": False,
                "latent_weights": True,
                "running_stats": False,
                "affine": False,
                "device": DEVICE,
                # "binary": True,
                # "freeze": True,
            },
            "training_parameters": {
                'n_epochs': 100,
                'batch_size': 256,
                'resize': False,
                "test_mcmc_samples": 1,
            },
            "criterion": torch.functional.F.nll_loss,
            # "optimizer": torch.optim.SGD,
            # "optimizer_parameters": {
            #     "lr": 0.001,
            #     "momentum": 0.9,
            #     "weight_decay": 0.0005,
            # },
            "optimizer": BinaryHomosynapticUncertaintyTest,
            "optimizer_parameters": {
                "lr": 0.1,
                "scale": 0,
                "gamma": 0,
                "noise": 0,
                "quantization": None,
                "threshold": None,
                "update": 1,
                "num_mcmc_samples": 1,
            },
            "task": "CIFAR100",
            "n_tasks": 1,  # When "PermutedMNIST" is selected, this parameter is the number of tasks
            "n_classes": 100,
        }
    ]

    for index, data in enumerate(networks_data):
        ### NAME INITIALIZATION ###
        name = f"{data['nn_type'].__name__}-{data['optimizer'].__name__}-{'-'.join([str(layer) for layer in data['nn_parameters']['features']]) if 'features' in data['nn_parameters'] else 'UKNW'}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers']])}-{data['task']}-{data['nn_parameters']['activation_function'].__name__}"
        # add some parameters to the name
        for key, value in data['optimizer_parameters'].items():
            name += f"-{key}-{value}"
        print(f"Training {name}...")

        ### FOLDER INITIALIZATION ###
        main_folder = os.path.join(SAVE_FOLDER, name)

        ### ACCURACY INITIALIZATION ###
        accuracies = []
        batch_size = data['training_parameters']['batch_size']
        resize = data['training_parameters']['resize'] if "resize" in data["training_parameters"] else False
        train_loader, test_loader, shape, target_size = task_selection(loader=loader,
                                                                       task=data["task"],
                                                                       n_tasks=data["n_tasks"],
                                                                       batch_size=batch_size,
                                                                       resize=resize)
        if "features" in data['nn_parameters']:
            # add input size to the layer of the network parameters
            data['nn_parameters']['features'].insert(0, shape[0])
        # add output size to the layer of the network parameters
        data['nn_parameters']['layers'].append(target_size)

        ### FOR EACH NETWORK IN THE DICT ###
        for iteration in range(N_NETWORKS):
            ### INIT NETWORK ###
            if iteration != 0:
                torch.manual_seed(SEED + iteration)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(SEED + iteration)
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

            # Setting the task iterator
            task_iterator = None
            if data["task"] == "PermutedMNIST":
                permutations = [torch.randperm(torch.prod(torch.tensor(shape)))
                                for _ in range(data["n_tasks"])]
                task_iterator = enumerate(
                    train_loader[0].permute_dataset(permutations))
            elif "CIFAR" in data["task"] and data["n_tasks"] > 1:
                task_iterator = enumerate(
                    train_loader[0].class_incremental_dataset(n_tasks=data["n_tasks"], n_classes=data["n_classes"]))
            else:
                task_iterator = enumerate(train_loader)

            for i, task in task_iterator:
                pbar = tqdm.trange(data["training_parameters"]["n_epochs"])
                for epoch in pbar:
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
                    # update the progress bar
                    name_loader = None
                    if data["task"] == "PermutedMNIST":
                        net_trainer.evaluate(test_loader[0].permute_dataset(
                            permutations))
                    elif "CIFAR" in data["task"] and data["n_tasks"] > 1:
                        net_trainer.evaluate(test_loader[0].class_incremental_dataset(
                            n_tasks=data["n_tasks"], n_classes=data["n_classes"], test=True))
                        name_loader = [f"Classes {i}-{i+data['n_classes']-1}" for i in range(
                            0, data["n_tasks"]*data["n_classes"], data["n_classes"])]
                    else:
                        net_trainer.evaluate(test_loader)

                    net_trainer.pbar_update(
                        pbar, epoch, data["training_parameters"]["n_epochs"], name_loader=name_loader)

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

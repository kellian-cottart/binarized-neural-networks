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
N_NETWORKS = 5  # Number of networks to train
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Number of workers for data loading when using CPU
PADDING = 2


### PATHS ###
SAVE_FOLDER = "saved_deep_models"
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
                ### NETWORK ###
                "layers": [2048, 2048],
                "device": DEVICE,
                "dropout": False,
                "init": "uniform",
                "std": 0.1,
                "bias": False,
                "latent_weights": True,
                "activation_function": Sign.apply,
                # "activation_function": torch.functional.F.relu,
                "output_function": "log_softmax",
                ### NORMALIZATION ###
                "normalization": "batchnorm",
                "momentum": 0,
                "eps": 0,
                "running_stats": False,
                "affine": False,
                "gnnum_groups": 1,
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 512,
                "test_mcmc_samples": 1,
                'resize': True,
            },
            "criterion": torch.functional.F.nll_loss,
            # "optimizer": BayesBiNN,
            # "optimizer_parameters": {
            #     "lr": 0.005,
            #     "temperature": 1,
            #     "scale": 0.1,
            #     "num_mcmc_samples": 1
            # },
            # "optimizer": torch.optim.Adam,
            # "optimizer_parameters": {
            #     "lr": 0.001,
            # },
            # "optimizer": MetaplasticAdam,
            # "optimizer_parameters": {
            #     "lr": 0.05,
            #     "metaplasticity": 1.2,
            # },
            "optimizer": BinaryHomosynapticUncertaintyTest,
            "optimizer_parameters": {
                "lr": 0.025,
                "scale": 0.6,
                "gamma": 0,
                "update": 5,
                "num_mcmc_samples": samples,
                "init_lambda": init,
            },
            "task": "CILCIFAR100",
            "n_tasks": 2,
            "n_classes": 50,
            "n_subsets": 1,
            "show_train": False,
        } for init in [0.01, 0.1, 1] for samples in [1, 2, 5, 10]
    ]

    for index, data in enumerate(networks_data):

        ### NAME INITIALIZATION ###
        name = f"{data['optimizer'].__name__}-BS{data['training_parameters']['batch_size']}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers']])}-{data['task']}-{data['nn_parameters']['activation_function'].__name__}-{data['nn_parameters']['normalization']}"
        # add some parameters to the name
        for key, value in data['optimizer_parameters'].items():
            name += f"-{key}-{value}"
        # add the number of epochs
        name += f"-n_epochs={'-'.join([str(epoch) for epoch in data['training_parameters']['n_epochs']])}" if isinstance(
            data['training_parameters']['n_epochs'], list) else f"-n_epochs={data['training_parameters']['n_epochs']}"
        if "n_tasks" in data:
            name += f"-n_tasks={data['n_tasks']}"
        if "n_classes" in data:
            name += f"-n_classes={data['n_classes']}"
        print(f"Training {name}...")

        ### FOLDER INITIALIZATION ###
        main_folder = os.path.join(SAVE_FOLDER, name)

        ### ACCURACY INITIALIZATION ###
        accuracies = []
        batch_size = data['training_parameters']['batch_size']
        resize = data['training_parameters']['resize'] if "resize" in data["training_parameters"] else False
        ### RELOADING DATASET ###
        train_loader, test_loader, shape, target_size = task_selection(loader=loader,
                                                                       task=data["task"],
                                                                       n_tasks=data["n_tasks"],
                                                                       batch_size=batch_size,
                                                                       resize=resize)
        # add input size to the layer of the network parameters
        data['nn_parameters']['layers'].insert(
            0, torch.prod(torch.tensor(shape)))
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

            ### TASK SELECTION ###
            # Setting the task iterator: The idea is that we yield datasets corresponding to the framework we want to use
            # For example, if we want to use the permuted framework, we will yield datasets with permuted images, not dependant on the dataset
            task_iterator = None
            if "Permuted" in data["task"]:
                # Create n_tasks permutations of the datasetc
                permutations = [torch.randperm(torch.prod(torch.tensor(shape)))
                                for _ in range(data["n_tasks"])]
                task_iterator = train_loader[0].permute_dataset(permutations)
            elif "CIL" in data["task"]:
                # Create n_tasks subsets of n_classes (need to permute the selection of classes for each task)
                rand = torch.randperm(target_size)
                # n_tasks subsets of n_classes without overlapping
                permutations = [rand[i:i+data["n_classes"]]
                                for i in range(0, target_size, data["n_classes"])]
                task_iterator = train_loader[0].class_incremental_dataset(
                    permutations=permutations)
            elif "Stream" in data["task"]:
                # Split the dataset in n_tasks subsets
                task_iterator = train_loader[0].stream_dataset(
                    data["n_subsets"])
            else:
                task_iterator = train_loader

            ### TRAINING ###
            for i, task in enumerate(task_iterator):
                epochs = data["training_parameters"]["n_epochs"][i] if isinstance(
                    data["training_parameters"]["n_epochs"], list) else data["training_parameters"]["n_epochs"]
                pbar = tqdm.trange(epochs)
                for epoch in pbar:
                    net_trainer.epoch_step(task)
                    ### TEST EVALUATION ###
                    # Depending on the task, we also need to use the framework on the test set and show training or not
                    name_loader = None
                    if "Permuted" in data["task"]:
                        predicted, certainty = net_trainer.evaluate(test_loader[0].permute_dataset(
                            permutations), train_loader=train_loader[0].permute_dataset(permutations) if "show_train" in data and data["show_train"] else None)
                    elif "CIL" in data["task"]:
                        predicted, certainty = net_trainer.evaluate(test_loader[0].class_incremental_dataset(
                            permutations=permutations), train_loader=train_loader[0].class_incremental_dataset(permutations) if "show_train" in data and data["show_train"] else None)
                        name_loader = [f"Classes {i}-{i+data['n_classes']-1}" for i in range(
                            0, data["n_tasks"]*data["n_classes"], data["n_classes"])]
                    else:
                        predicted, certainty = net_trainer.evaluate(
                            test_loader, train_loader=train_loader if "show_train" in data and data["show_train"] else None)
                    # update the progress bar
                    net_trainer.pbar_update(
                        pbar, epoch, data["training_parameters"]["n_epochs"], name_loader=name_loader)

                    ### EXPORT VISUALIZATION OF PARAMETERS ###
                    if epoch % 20 == 0:
                        # print predicted and associated certainty
                        visualize_certainty(
                            predicted=predicted,
                            certainty=certainty,
                            log=True,

                            path=os.path.join(main_folder, "certainty"),
                        )
                        if data["optimizer"] in [BinaryHomosynapticUncertainty, BinaryHomosynapticUncertaintyTest]:
                            visualize_lambda(
                                parameters=net_trainer.optimizer.param_groups[0]['params'],
                                lambda_=net_trainer.optimizer.state['lambda'],
                                path=os.path.join(main_folder, "lambda"),
                                threshold=10,
                                task=i,
                                epoch=epoch,
                            )
                            visualize_grad(
                                parameters=net_trainer.optimizer.param_groups[0]['params'],
                                grad=net_trainer.optimizer.state['lrgrad'],
                                path=os.path.join(main_folder, "lrgrad"),
                                threshold=1e-1,
                                task=i,
                                epoch=epoch,
                            )
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
        visualize_sequential(title, accuracies, folder=main_folder,
                             epochs=data["training_parameters"]["n_epochs"])
        # if number of tasks is 100, export the accuracy of the first 10 and last 10 tasks
        if data['n_tasks'] >= 10:
            title = name + "-tasks-1-10"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=1, t_end=10)
        if data['n_tasks'] == 100:
            title = name + "-tasks-91-100"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=91, t_end=100)

from utils import *
from dataloader import *
import models
import trainer
from optimizers import *
import os
import json
import datetime
from torch import device, cuda, functional, stack, save, prod, set_default_device, set_default_dtype, manual_seed, randperm
from math import sqrt
from torch.optim import SGD, Adam
import tqdm
from numpy.random import seed as npseed

SEED = 1000  # Random seed
N_NETWORKS = 1  # Number of networks to train
DEVICE = device("cuda:0")
GRAPHS = True
PBAR = False
MODULO = 10
### PATHS ###
SAVE_FOLDER = "saved_deep_models"
DATASETS_PATH = "datasets"

if __name__ == "__main__":
    ### SEED ###
    npseed(SEED)
    manual_seed(SEED)
    cuda.manual_seed(SEED)
    set_default_device(DEVICE)
    set_default_dtype(float32)
    ### INIT DATALOADER ###
    loader = GPULoading(device=DEVICE, root=DATASETS_PATH)
    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "image_padding": 0,
            "nn_type": models.ResNet18Bayesian,
            "nn_parameters": {
                # NETWORK ###
                "layers": [],
                # "features": [16, 32, 64],
                "kernel_size": [3, 3, 3],
                "padding": "same",
                "device": DEVICE,
                "dropout": False,
                "bias": False,
                "n_samples_test": 5,
                "n_samples_train": 5,
                "tau": 1,
                "std": sqrt(1e-4),
                "activation_function": "relu",
                "activation_parameters": {
                    "width": 1,
                    "power": 2,
                },
                "normalization": "",
                "eps": 1e-5,
                "momentum": 0.15,
                "running_stats": False,
                "affine": False,
                "frozen": False,
                "sigma_multiplier": 1,
                "version": 0,
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 128,
                'test_batch_size': 128,
                'feature_extraction': False,
                'data_aug_it': 1,
                'full': False,
                "continual": False,
                "task_boundaries": False,
            },
            "label_trick": False,
            "output_function": "log_softmax",
            "criterion": functional.F.nll_loss,
            "reduction": "sum",
            "optimizer": MESU,
            "optimizer_parameters": {
                "sigma_prior": sqrt(1e-2),
                "mu_prior": 0,
                "N_mu": 1_000_000,
                "N_sigma": 1_000_000,
                "lr_mu": 1,
                "lr_sigma": 1,
                "norm_term": False,
            },
            # "optimizer": BHUparallel,
            # "optimizer_parameters": {
            #     "lr_max": 5,
            #     "ratio_coeff": 1,
            #     "metaplasticity": 1,
            # },
            # "optimizer": SGD,
            # "optimizer_parameters": {
            #     "lr": 0.0001,
            # },
            # "optimizer": MESUDET,
            # "optimizer_parameters": {
            #     "mu_prior": 0,
            #     "sigma_prior": 0.1,
            #     "N_mu": 1_000_000,
            #     "N_sigma": 1_000_000,
            #     "c_sigma": 1,
            #     "c_mu": 1,
            #     "second_order": True,
            #     "clamp_sigma": [0, 0],
            #     "clamp_mu": [0, 0],
            #     "enforce_learning_sigma": False,
            #     "normalise_grad_sigma": 0,
            #     "normalise_grad_mu": 0,
            #     "noise_variance": 0,
            # },
            "task": "DILCIFAR100",
            "n_tasks": 5,
            "n_classes": 1,
        }
    ]

    for index, data in enumerate(networks_data):
        RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")
        ### FOLDER INITIALIZATION ###
        name = f"{data['optimizer'].__name__}-" + f"{data['nn_type'].__name__}" + \
            f"-BS{data['training_parameters']['batch_size']}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers']])}-{data['task']}-{data['nn_parameters']['activation_function']}"
        if "regularizer" in data:
            name += f"-{data['regularizer']['type']}"
        ### ACCURACY INITIALIZATION ###
        accuracies, training_accuracies = [], []
        batch_size = data['training_parameters']['batch_size']
        feature_extraction = data['training_parameters']['feature_extraction'] if "feature_extraction" in data["training_parameters"] else False
        data_aug_it = data['training_parameters']['data_aug_it'] if "data_aug_it" in data["training_parameters"] else None
        # add input/output size to the layer of the network parameters
        if "Conv" in data["nn_type"].__name__:
            name += "-".join([str(feature)
                             for feature in data['nn_parameters']['features']])
        ### MAIN FOLDER ###
        main_folder = os.path.join(SAVE_FOLDER, RUN_ID+name)
        for iteration in range(N_NETWORKS):
            ### SEEDING ###
            manual_seed(SEED + iteration)
            cuda.manual_seed(SEED + iteration)
            ### LOADING DATASET ###
            train_dataset, test_dataset, shape, target_size = loader.task_selection(
                task=data["task"], n_tasks=data["n_tasks"], batch_size=batch_size, feature_extraction=feature_extraction, iterations=data_aug_it, padding=data["image_padding"], run=iteration, full=data["training_parameters"]["full"] if "full" in data["training_parameters"] else False)
            if iteration == 0:
                data['nn_parameters']['layers'].append(target_size)

                conv_type = ["VGG", "ResNet", "EfficientNet", "cifar"]
                if not any([conv.lower() in data["nn_type"].__name__.lower() for conv in conv_type]):
                    data['nn_parameters']['layers'].insert(
                        0, prod(tensor(shape)))

            # Instantiate the network
            model = data['nn_type'](**data['nn_parameters'])
            print(model)
            ### INSTANTIATE THE TRAINER ###
            if data["optimizer"] in [BayesBiNN]:
                net_trainer = trainer.BayesTrainer(batch_size=batch_size,
                                                   model=model, **data, device=DEVICE)
            else:
                net_trainer = trainer.GPUTrainer(batch_size=batch_size,
                                                 model=model, **data, device=DEVICE)
            if isinstance(net_trainer.optimizer, MetaplasticAdam) and net_trainer.model.affine and data["training_parameters"]["task_boundaries"]:
                batch_params = []
                for i in range(data["n_tasks"]):
                    batch_params.append(net_trainer.model.save_bn_states())
            ### CREATING PERMUTATIONS ###
            permutations = None
            if "PermutedLabels" in data["task"]:
                permutations = [randperm(target_size)
                                for _ in range(data["n_tasks"])]
            elif "Permuted" in data["task"]:
                permutations = [randperm(prod(tensor(shape)))
                                for _ in range(data["n_tasks"])]
            if "CIL" in data["task"]:
                # Create the permutations for the class incremental scenario: n_classes per task with no overlap
                random_permutation = randperm(target_size)
                if isinstance(data["n_classes"], int):
                    permutations = [random_permutation[i * data["n_classes"]:(i + 1) * data["n_classes"]]
                                    for i in range(data["n_tasks"])]
                else:
                    # n_classes in a list of number of class to take per permutation, we want them to not overlap
                    permutations = [random_permutation[sum(data["n_classes"][:i]):sum(data["n_classes"][:i+1])]
                                    for i in range(data["n_tasks"])]
                # Create GPUTensordataset with only the class in each permutation
                if not isinstance(train_dataset, list):
                    train_dataset = [train_dataset.__getclasses__(permutation)
                                     for permutation in permutations]
                    test_dataset = [test_dataset.__getclasses__(permutation)
                                    for permutation in permutations]
            test_dataset = test_dataset if isinstance(
                test_dataset, list) else [test_dataset]
            ### TASK SELECTION ###
            for i in range(data["n_tasks"]):
                epochs = data["training_parameters"]["n_epochs"][i] if isinstance(
                    data["training_parameters"]["n_epochs"], list) else data["training_parameters"]["n_epochs"]
                for epoch in range(epochs):
                    predictions, labels = net_trainer.epoch_step(batch_size=batch_size,
                                                                 test_batch_size=data["training_parameters"]["test_batch_size"],
                                                                 train_dataset=train_dataset,
                                                                 test_dataset=test_dataset,
                                                                 task_id=i,
                                                                 permutations=permutations,
                                                                 pbar=PBAR,
                                                                 epoch=epoch,
                                                                 epochs=epochs,
                                                                 continual=data["training_parameters"]["continual"],
                                                                 batch_params=batch_params if data["optimizer"] in [
                                                                     MetaplasticAdam] and net_trainer.model.affine and data["training_parameters"]["task_boundaries"] else None)
                    ### EXPORT VISUALIZATION OF PARAMETERS ###
                    if GRAPHS:
                        graphs(main_folder=main_folder, net_trainer=net_trainer, task=i,
                               n_tasks=data["n_tasks"], epoch=epoch, predictions=predictions, labels=labels, modulo=MODULO)
                ### TASK BOUNDARIES ###
                if data["training_parameters"]["task_boundaries"] == True and isinstance(net_trainer.optimizer, BayesBiNN):
                    net_trainer.optimizer.update_prior_lambda()
            ### SAVING DATA ###
            os.makedirs(main_folder, exist_ok=True)
            sub_folder = os.path.join(
                main_folder, f"params-network-{iteration}")
            os.makedirs(sub_folder, exist_ok=True)
            net_trainer.save(os.path.join(sub_folder, "weights.pt"))
            string = json.dumps(data, default=lambda x: x.__name__ if hasattr(
                x, "__name__") else str(x), indent=4)
            with open(os.path.join(sub_folder, "config.json"), "w") as f:
                f.write(string)
            save(stack(net_trainer.testing_accuracy),
                 os.path.join(sub_folder, "accuracy.pt"))
            accuracies.append(stack(net_trainer.testing_accuracy))
            if "training_accuracy" in dir(net_trainer) and len(net_trainer.training_accuracy) > 0:
                training_accuracies.append(
                    stack(net_trainer.training_accuracy))
                save(stack(net_trainer.training_accuracy),
                     os.path.join(sub_folder, "training_accuracy.pt"))
        accuracies = stack(accuracies)
        if len(training_accuracies) > 0:
            training_accuracies = stack(training_accuracies)
        ### SAVE GRAPHS ###
        title = "tasks"
        visualize_sequential(title,
                             accuracies,
                             folder=main_folder,
                             epochs=data["training_parameters"]["n_epochs"],
                             training_accuracies=training_accuracies if len(
                                 training_accuracies) > 0 else None)

        # if number of tasks is 100, export the accuracy of the first 10 and last 10 tasks
        if data['n_tasks'] >= 10:
            title = "tasks-1-10"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=1, t_end=10)
            title = f"tasks-{data['n_tasks']-9}-{data['n_tasks']}"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=data['n_tasks']-9, t_end=data['n_tasks'])

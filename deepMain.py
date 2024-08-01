import models.midvggbibayesian
from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizers import *
import os
import json
import tqdm
import datetime

SEED = 1000  # Random seed
N_NETWORKS = 1  # Number of networks to train
DEVICE = torch.device("cuda:0")
GRAPHS = True
MODULO = 10
### PATHS ###
SAVE_FOLDER = "saved_deep_models"
DATASETS_PATH = "datasets"
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.set_default_device(DEVICE)
    torch.set_default_dtype(torch.float32)
    ### INIT DATALOADER ###
    loader = GPULoading(device=DEVICE, root=DATASETS_PATH)
    ### NETWORK CONFIGURATION ###
    networks_data = [
        {
            "image_padding": 2,
            "nn_type": models.ConvBayesianNeuralNetwork,
            "nn_parameters": {
                # NETWORK ###
                "layers": [4096],
                "features": [64, 128, 256],
                "kernel_size": [3, 3, 3],
                "padding": "same",
                "device": DEVICE,
                "dropout": False,
                "init": "gaussian",
                "std": 0.01,
                "n_samples_forward": 10,
                "n_samples_backward": 10,
                "tau": 1,
                "activation_function": "re",
                "activation_parameters": {
                    "width": 1,
                    "power": 4
                },
                "normalization": "instancenorm",
                "eps": 1e-5,
                "momentum": 0,
                "running_stats": False,
                "affine": False,
                "bias": False,
            },
            "training_parameters": {
                'n_epochs': 5,
                'batch_size': 512,
                'test_batch_size': 512,
                'feature_extraction': True,
                'data_aug_it': 1,
                "continual": True,
                "task_boundaries": False,
                "test_mcmc_samples": 10,
            },
            "label_trick": True,
            "output_function": "log_softmax",
            "criterion": torch.functional.F.nll_loss,
            "reduction": "sum",
            # "optimizer": BHUparallel,
            # "optimizer_parameters": {
            #     "lr_max": 1000,
            #     "metaplasticity": 3,
            #     "ratio_coeff": 0.1,
            #     "mesuified": False,
            #     "N": 20_000,
            #     "normalize_gradients": False,
            # },
            "optimizer": MESU,
            "optimizer_parameters": {
                "sigma_prior": 0.01,
                "N": 1e5,
                "clamp_grad": 0,
            },
            # "optimizer": BayesBiNN,
            # "optimizer_parameters": {
            #     "train_set_size": 10000,
            #     "betas": 0.0,
            #     "lr": 1e-5,
            #     "prior_lambda": None,
            #     "num_samples": 10,
            #     "temperature": 1e-5,
            #     "reweight": 0,
            # },
            # "optimizer": MetaplasticAdam,
            # "optimizer_parameters": {"lr": 0.008, "metaplasticity": 3},
            # "optimizer": torch.optim.SGD,
            # "optimizer_parameters": {"lr": 0.01, "momentum": 0.1},
            "task": "MNIST",
            "n_tasks": 1,
            "n_classes": 1,
        }
    ]

    for index, data in enumerate(networks_data):
        ### FOLDER INITIALIZATION ###
        name = f"{data['optimizer'].__name__}-BS{data['training_parameters']['batch_size']}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers']])}-{data['task']}-{data['nn_parameters']['activation_function']}"
        ### ACCURACY INITIALIZATION ###
        accuracies, training_accuracies = [], []
        batch_size = data['training_parameters']['batch_size']
        feature_extraction = data['training_parameters']['feature_extraction'] if "feature_extraction" in data["training_parameters"] else False
        data_aug_it = data['training_parameters']['data_aug_it'] if "data_aug_it" in data["training_parameters"] else None
        ### LOADING DATASET ###
        train_dataset, test_dataset, shape, target_size = loader.task_selection(
            task=data["task"], n_tasks=data["n_tasks"], batch_size=batch_size, feature_extraction=feature_extraction, iterations=data_aug_it, padding=data["image_padding"])
        # add input/output size to the layer of the network parameters
        if "Conv" in data["nn_type"].__name__:
            name += "-conv-" + \
                "-".join([str(feature)
                         for feature in data['nn_parameters']['features']])
            data['nn_parameters']['features'].insert(
                0, shape[0])
        elif not "VGG" in data["nn_type"].__name__:
            data['nn_parameters']['layers'].insert(
                0, torch.prod(torch.tensor(shape)))
        data['nn_parameters']['layers'].append(target_size)
        ### MAIN FOLDER ###
        main_folder = os.path.join(SAVE_FOLDER, RUN_ID+name)
        for iteration in range(N_NETWORKS):
            ### INIT NETWORK ###
            torch.manual_seed(SEED + iteration)
            torch.cuda.manual_seed(SEED + iteration)
            # Instantiate the network
            model = data['nn_type'](**data['nn_parameters'])
            ### INSTANTIATE THE TRAINER ###
            if data["optimizer"] in [BayesBiNN]:
                net_trainer = trainer.BayesTrainer(batch_size=batch_size,
                                                   model=model, **data, device=DEVICE)
            else:
                net_trainer = trainer.GPUTrainer(batch_size=batch_size,
                                                 model=model, **data, device=DEVICE)
            print(net_trainer.model)
            if data["optimizer"] in [MetaplasticAdam] and net_trainer.model.affine:
                batch_params = []
                for i in range(data["n_tasks"]):
                    batch_params.append(net_trainer.model.save_bn_states())

            ### CREATING PERMUTATIONS ###
            permutations = None
            if "Permuted" in data["task"]:
                permutations = [torch.randperm(torch.prod(torch.tensor(shape)))
                                for _ in range(data["n_tasks"])]
            if "CIL" in data["task"]:
                # Create the permutations for the class incremental scenario: n_classes per task with no overlap
                random_permutation = torch.randperm(target_size)
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
                task_train_dataset = train_dataset[i] if isinstance(
                    train_dataset, list) else train_dataset
                pbar = tqdm.trange(epochs)
                for epoch in pbar:
                    num_batches = len(task_train_dataset) // batch_size + 1
                    task_train_dataset.shuffle()
                    for n_batch in range(num_batches):
                        ### TRAINING ###
                        batch, labels = batch_yielder(
                            dataset=task_train_dataset,
                            task=data["task"],
                            batch_size=batch_size,
                            task_id=i,
                            iteration=n_batch,
                            max_iterations=epochs*num_batches,
                            permutations=permutations,
                            epoch=epoch,
                            continual=data["training_parameters"]["continual"] if "continual" in data["training_parameters"] else None
                        )
                        net_trainer.batch_step(batch, labels)
                    if data["optimizer"] in [MetaplasticAdam] and net_trainer.model.affine:
                        batch_params[i] = net_trainer.model.save_bn_states()
                    ### TESTING ###
                    # Depending on the task, we also need to use the framework on the test set and show training or not
                    predictions, labels = evaluate_tasks(
                        dataset=test_dataset,
                        task=data["task"],
                        net_trainer=net_trainer,
                        permutations=permutations,
                        batch_size=data["training_parameters"]["test_batch_size"],
                        train_dataset=task_train_dataset,
                        batch_params=batch_params if data["optimizer"] in [
                            MetaplasticAdam] and net_trainer.model.affine else None
                    )
                    net_trainer.pbar_update(
                        pbar, epoch, n_epochs=epochs, n_tasks=data["n_tasks"], task=i)
                    if data["optimizer"] in [MetaplasticAdam] and net_trainer.model.affine:
                        net_trainer.model.load_bn_states(batch_params[i])
                    ### EXPORT VISUALIZATION OF PARAMETERS ###
                    if GRAPHS:
                        graphs(main_folder=main_folder, net_trainer=net_trainer, task=i,
                               n_tasks=data["n_tasks"], epoch=epoch, predictions=predictions, labels=labels, modulo=MODULO)
                ### TASK BOUNDARIES ###
                if data["training_parameters"]["task_boundaries"] == True:
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
            torch.save(net_trainer.testing_accuracy,
                       os.path.join(sub_folder, "accuracy.pt"))
            accuracies.append(torch.stack(net_trainer.testing_accuracy))
            if "training_accuracy" in dir(net_trainer) and len(net_trainer.training_accuracy) > 0:
                torch.save(net_trainer.training_accuracy,
                           os.path.join(sub_folder, "training_accuracy.pt"))
                training_accuracies.append(
                    torch.stack(net_trainer.training_accuracy))
        accuracies = torch.stack(accuracies)
        ### SAVE GRAPHS ###
        title = "tasks"
        visualize_sequential(title,
                             accuracies,
                             folder=main_folder,
                             epochs=data["training_parameters"]["n_epochs"],
                             training_accuracies=training_accuracies if "show_train" in data and data[
                                 "show_train"] else None)

        # if number of tasks is 100, export the accuracy of the first 10 and last 10 tasks
        if data['n_tasks'] >= 10:
            title = "tasks-1-10"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=1, t_end=10)
            title = f"tasks-{data['n_tasks']-9}-{data['n_tasks']}"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=data['n_tasks']-9, t_end=data['n_tasks'])

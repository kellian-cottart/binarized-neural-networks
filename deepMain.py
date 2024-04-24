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
PADDING = 2
GRAPHS = True

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
                # NETWORK ###w
                "layers": [2048, 2048],
                "device": DEVICE,
                "dropout": False,
                "init": "uniform",
                "std": 0.1,
                "bias": False,
                "activation_function": Sign.apply,
                # "activation_function": torch.functional.F.relu,
                "output_function": "log_softmax",
                ### NORMALIZATION ###
                "normalization": "instancenorm",
                "running_stats": False,
                "affine": False,
                "gnnum_groups": 1,
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 128,
                "test_mcmc_samples": 10,
                'resize': True,
            },
            "criterion": torch.functional.F.nll_loss,
            # "optimizer": BinaryHomosynapticUncertaintyTest,
            # "optimizer_parameters": {
            #     "lr": 7,
            #     "beta": 0.7,
            #     "gamma": 0.2,
            #     "num_mcmc_samples": 1,
            #     "init_law": "gaussian",
            #     "init_param": 1,
            # },
            # "optimizer": MetaplasticSGD,
            # "optimizer_parameters": {
            #     "lr": 400,
            #     "beta": 0,
            #     "gamma": 0,
            # },
            "optimizer": MetaplasticAdam,
            "optimizer_parameters": {
                "lr": 0.025,
                "metaplasticity": 1.3,
            },
            "task": "CILCIFAR100",
            "n_tasks": 2,
            "n_classes": 50,
            "n_subsets": 1,
            "n_repetition": 1,
            "show_train": True,
        }
    ]
    for index, data in enumerate(networks_data):
        ### FOLDER INITIALIZATION ###
        name = f"{data['optimizer'].__name__}-BS{data['training_parameters']['batch_size']}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers']])}-{data['task']}-{data['nn_parameters']['activation_function'].__name__}"
        name = versionning(SAVE_FOLDER, name, "")
        print(f"Training {name}...")
        main_folder = name

        ### ACCURACY INITIALIZATION ###
        accuracies, training_accuracies = [], []
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
            if data["optimizer"] in [BayesBiNN, BinaryHomosynapticUncertaintyTest]:
                net_trainer = trainer.BayesTrainer(batch_size=batch_size,
                                                   model=model, **data, device=DEVICE)
            else:
                net_trainer = trainer.GPUTrainer(batch_size=batch_size,
                                                 model=model, **data, device=DEVICE)
            print(net_trainer.model)
            for k in range(data["n_repetition"]):
                ### TASK SELECTION ###
                # Setting the task iterator: The idea is that we yield datasets corresponding to the framework we want to use
                # For example, if we want to use the permuted framework, we will yield datasets with permuted images, not dependant on the dataset
                task_iterator, permutations = iterable_selector(
                    data, train_loader, shape, target_size)
                for i, task in enumerate(task_iterator):
                    epochs = data["training_parameters"]["n_epochs"][i + k * data["n_tasks"]] if isinstance(
                        data["training_parameters"]["n_epochs"], list) else data["training_parameters"]["n_epochs"]
                    pbar = tqdm.trange(epochs)
                    for epoch in pbar:
                        ### TRAINING ###
                        net_trainer.epoch_step(task)
                        ### TESTING ###
                        # Depending on the task, we also need to use the framework on the test set and show training or not
                        name_loader, predictions, labels = iterable_evaluation_selector(
                            data, train_loader, test_loader, net_trainer, permutations)
                        net_trainer.pbar_update(
                            pbar, epoch, data["training_parameters"]["n_epochs"], name_loader=name_loader)
                        ### EXPORT VISUALIZATION OF PARAMETERS ###
                        if GRAPHS:
                            graphs(data, main_folder, net_trainer,
                                   i, epoch, predictions, labels)
                    ### TASK BOUNDARIES ###
                    if data["optimizer"] in [BayesBiNN]:
                        net_trainer.optimizer.update_prior_lambda()
                        ### SAVING DATA ###
            os.makedirs(main_folder, exist_ok=True)
            sub_folder = os.path.join(
                main_folder, f"params-network-{iteration}")
            os.makedirs(sub_folder, exist_ok=True)
            ### SAVE WEIGHTS ###
            net_trainer.save(os.path.join(sub_folder, "weights.pt"))
            ### SAVE CONFIGURATION ###
            # there are classes in the data, so we need to convert them to string
            string = json.dumps(data, default=lambda x: str(x))
            with open(os.path.join(sub_folder, "config.json"), "w") as f:
                f.write(string)
            ### SAVE ACCURACY ###
            torch.save(net_trainer.testing_accuracy,
                       os.path.join(sub_folder, "accuracy.pt"))
            accuracies.append(net_trainer.testing_accuracy)
            if "training_accuracy" in dir(net_trainer):
                torch.save(net_trainer.training_accuracy,
                           os.path.join(sub_folder, "training_accuracy.pt"))
                training_accuracies.append(net_trainer.training_accuracy)
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
        if data['n_tasks'] == 100:
            title = "tasks-91-100"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=91, t_end=100)
 

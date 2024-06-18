from utils import *
from dataloader import *
import models
import torch
import trainer
from optimizers import *
import os
import json
import tqdm

SEED = 1000  # Random seed
N_NETWORKS = 1  # Number of networks to train
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Number of workers for data loading when using CPU
PADDING = 2
GRAPHS = True
MODULO = 10
### PATHS ###
SAVE_FOLDER = "saved_deep_models"
DATASETS_PATH = "datasets"
HESSIAN_COMP = False

if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)
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
            "nn_type": models.BiBayesianNN,
            "nn_parameters": {
                # NETWORK ###w
                "layers": [1024],
                # "features": [32, 64, 128],
                # "kernel_size": 3,
                # "padding": "same",
                # "stride": 1,
                "device": DEVICE,
                "dropout": False,
                "init": "gaussian",
                "std": 0.05,
                "n_samples_forward": 10,
                "n_samples_backward": 10,
                "tau": 1,
                "binarized": False,
                "squared_inputs": False,
                "activation_function": "signelephant",
                "output_function": "log_softmax",
                "normalization": "instancenorm",
                "eps": 1e-5,
                "momentum": 0,
                "running_stats": False,
                "affine": False,
                "bias": False,
            },
            "training_parameters": {
                'n_epochs': 20,
                'batch_size': 128,
                'resize': True,
                'data_aug_it': 1,
                "continual": False,
                "task_boundaries": False,
            },
            "criterion": torch.functional.F.nll_loss,
            "reduction": "sum",
            "optimizer": BHUparallel,
            "optimizer_parameters": {
                "lr_mult": 6,
                "lr_max": 2,
                "likelihood_coeff": 1,
                "kl_coeff": 1,
                "normalize_gradients": False,
                "eps": 1e-7,
                "clamp": 0.1,
                "mesuified": False,
                "N": 10_000,
            },
            # "optimizer": BayesBiNNParallel,
            # "optimizer_parameters": {
            #     "lr": 50,
            #     "clamp_cosh": 20,
            #     "beta": 0,
            #     "scale": 0,
            # },
            # "optimizer": torch.optim.Adam,
            # "optimizer_parameters": {
            #     "lr": 0.001,
            # },
            # "optimizer": MetaplasticAdam,
            # "optimizer_parameters": {
            #     "lr": 0.005,
            #     "metaplasticity": 1.35
            # },
            "task": "PermutedMNIST",
            "n_tasks": 10,
            "n_classes": 1,
            "n_repetition": 1,
        }
    ]

    for index, data in enumerate(networks_data):
        ### FOLDER INITIALIZATION ###
        name = f"{data['optimizer'].__name__}-BS{data['training_parameters']['batch_size']}-{'-'.join([str(layer) for layer in data['nn_parameters']['layers']])}-{data['task']}-{data['nn_parameters']['activation_function']}"
        ### ACCURACY INITIALIZATION ###
        accuracies, training_accuracies = [], []
        batch_size = data['training_parameters']['batch_size']
        resize = data['training_parameters']['resize'] if "resize" in data["training_parameters"] else False
        data_aug_it = data['training_parameters']['data_aug_it'] if "data_aug_it" in data["training_parameters"] else None

        ### LOADING DATASET ###
        train_dataset, test_dataset, shape, target_size = task_selection(
            loader=loader,
            task=data["task"],
            n_tasks=data["n_tasks"],
            batch_size=batch_size,
            resize=resize,
            iterations=data_aug_it
        )

        ### CREATING PERMUTATIONS ###
        permutations = None
        if "Permuted" in data["task"]:
            permutations = [torch.randperm(torch.prod(torch.tensor(shape)))
                            for _ in range(data["n_tasks"])]
        if "CIL" in data["task"]:
            # Create the permutations for the class incremental scenario: n_classes per task with no overlap
            random_permutation = torch.randperm(target_size)
            permutations = [random_permutation[i * data["n_classes"]                                               :(i + 1) * data["n_classes"]] for i in range(data["n_tasks"])]

        # add input/output size to the layer of the network parameters
        if "Conv" in data["nn_type"].__name__:  # Convolutional network
            name += "-conv-"
            name += "-".join([str(feature)
                              for feature in data['nn_parameters']['features']])
            data['nn_parameters']['features'].insert(
                0, shape[0])
        else:
            data['nn_parameters']['layers'].insert(
                0, torch.prod(torch.tensor(shape)))
        data['nn_parameters']['layers'].append(target_size)

        name = versionning(SAVE_FOLDER, name, "")
        print(f"Training {name}...")
        main_folder = name

        for iteration in range(N_NETWORKS):
            ### INIT NETWORK ###
            if iteration != 0:
                torch.manual_seed(SEED + iteration)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(SEED + iteration)
            model = data['nn_type'](**data['nn_parameters'])
            ident = f"{name} - {index}"
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
            for k in range(data["n_repetition"]):
                ### TASK SELECTION ###
                # Setting the task iterator: The idea is that we yield datasets corresponding to the framework we want to use
                # For example, if we want to use the permuted framework, we will yield datasets with permuted images, not dependant on the dataset
                for i in range(data["n_tasks"]):
                    epochs = data["training_parameters"]["n_epochs"][i + k * data["n_tasks"]] if isinstance(
                        data["training_parameters"]["n_epochs"], list) else data["training_parameters"]["n_epochs"]
                    pbar = tqdm.trange(epochs)
                    for epoch in pbar:
                        num_batches = len(train_dataset) // (
                            batch_size * data["n_tasks"]) - 1 if "CIL" in data["task"] or "Stream" in data["task"] else len(train_dataset) // batch_size - 1
                        train_dataset.shuffle()
                        for n_batch in range(num_batches):
                            # if HESSIAN_COMP == True:
                            #     batch_hessians = []
                            #     batch_gradients = []
                            #     batch_hessians.append(
                            #         net_trainer.compute_hessian())
                            #     batch_gradients.append(
                            #         net_trainer.model.layers[-2].lambda_.grad)
                            ### TRAINING ###
                            batch, labels = special_task_selector(
                                data,
                                train_dataset,
                                batch_size=batch_size,
                                task_id=i,
                                iteration=n_batch,
                                max_iterations=epochs*num_batches,
                                permutations=permutations,
                                epoch=epoch,
                                continual=data["training_parameters"]["continual"] if "continual" in data["training_parameters"] else None
                            )
                            net_trainer.batch_step(batch, labels)
                        # if HESSIAN_COMP == True:
                        #     # Compute the mean hessian of the batch
                        #     net_trainer.hessian.append(
                        #         torch.sum(torch.stack(batch_hessians), dim=0))
                        #     net_trainer.gradient.append(
                        #         torch.sum(torch.stack(batch_gradients), dim=0))
                        #     # save hessian
                        #     os.makedirs('hessian', exist_ok=True)
                        #     torch.save(net_trainer.gradient,
                        #                versionning('hessian', 'grad', ".pt"))
                        #     torch.save(net_trainer.hessian,
                        #                versionning('hessian', 'hessian', ".pt"))
                        #     torch.save(net_trainer.model.layers[-2].lambda_,
                        #                versionning('hessian', 'lambda', ".pt"))
                        if data["optimizer"] in [MetaplasticAdam] and net_trainer.model.affine:
                            batch_params[i] = net_trainer.model.save_bn_states()
                        ### TESTING ###
                        # Depending on the task, we also need to use the framework on the test set and show training or not
                        name_loader, predictions, labels = iterable_evaluation_selector(
                            data=data,
                            test_dataset=test_dataset,
                            net_trainer=net_trainer,
                            permutations=permutations,
                            batch_params=batch_params if data["optimizer"] in [
                                MetaplasticAdam] and net_trainer.model.affine else None,
                            train_dataset=train_dataset)
                        net_trainer.pbar_update(
                            pbar, epoch, n_epochs=epochs, name_loader=name_loader)
                        if data["optimizer"] in [MetaplasticAdam] and net_trainer.model.affine:
                            net_trainer.model.load_bn_states(batch_params[i])
                        ### EXPORT VISUALIZATION OF PARAMETERS ###
                        if GRAPHS:
                            graphs(main_folder=main_folder,
                                   net_trainer=net_trainer,
                                   task=i,
                                   epoch=epoch,
                                   predictions=predictions,
                                   labels=labels,
                                   task_test_length=test_dataset.data.shape[0] if "Permuted" in data[
                                       "task"] else test_dataset.data.shape[0] // data["n_tasks"],
                                   modulo=MODULO)

                    ### TASK BOUNDARIES ###
                    if data["training_parameters"]["task_boundaries"] == True:
                        net_trainer.optimizer.update_prior_lambda()
            ### SAVING DATA ###
            os.makedirs(main_folder, exist_ok=True)
            sub_folder = os.path.join(
                main_folder, f"params-network-{iteration}")
            os.makedirs(sub_folder, exist_ok=True)
            net_trainer.save(os.path.join(sub_folder, "weights.pt"))
            string = json.dumps(data, default=lambda x: str(x))
            with open(os.path.join(sub_folder, "config.json"), "w") as f:
                f.write(string)
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
            title = f"tasks-{data['n_tasks']-9}-{data['n_tasks']}"
            visualize_task_frame(
                title, accuracies, folder=main_folder, t_start=data['n_tasks']-9, t_end=data['n_tasks'])

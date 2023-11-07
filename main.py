from utils import *
import models
import torch
import matplotlib.pyplot as plt
import os
import datetime
import time

### GLOBAL VARIABLES ###
SEED = 2506
BATCH_SIZE = 100
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-8
METAPLASTICITY = 1.5
N_EPOCHS = 10
STD = 0.1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### PATHS ###
SAVE_FOLDER = "saved"
DATASETS_PATH = "datasets"


def versionning(folder, title, format=".pdf"):
    os.makedirs(folder, exist_ok=True)
    # YYYY-MM-DD-title-version
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    version = 1
    # while there exists a file with the same name
    while os.path.exists(os.path.join(folder, f"{timestamp}-{title}-v{version}"+format)):
        version += 1
    versionned = os.path.join(folder, f"{timestamp}-{title}-v{version}")
    return versionned + format


def visualize_sequential(title, t_accuracy, folder):
    plt.figure()
    plt.plot(range(1, len(t_accuracy)+1), t_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['MNIST', 'Fashion-MNIST'])
    os.makedirs(folder, exist_ok=True)
    versionned = versionning(folder, title)
    plt.savefig(versionned, bbox_inches='tight')


if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)

    ### LOAD DATASETS ###
    mnist_train, mnist_test = mnist(DATASETS_PATH, BATCH_SIZE)
    fashion_mnist_train, fashion_mnist_test = fashion_mnist(
        DATASETS_PATH, BATCH_SIZE)

    all_test = {'mnist_test': mnist_test,
                'fashion_mnist_test': fashion_mnist_test}

    input_size = mnist_train.dataset.data.shape[1] * \
        mnist_train.dataset.data.shape[2]

    ### NETWORKS ###
    networks_data = {
        "DNN": {
            "model": models.DNN([input_size, 4096, 4096, 10], init='uniform', std=STD, device=DEVICE),
            "parameters": {'n_epochs': N_EPOCHS, 'learning_rate': LEARNING_RATE,
                           'weight_decay': WEIGHT_DECAY, **all_test},
            "accuracy": []
        },
        "BNN wout Meta": {
            "model": models.BNN([input_size, 4096, 4096, 10], init='uniform', std=STD, device=DEVICE),
            "parameters": {'n_epochs': N_EPOCHS, 'learning_rate': LEARNING_RATE,
                           'weight_decay': WEIGHT_DECAY, 'metaplasticity': 0, **all_test},
            "accuracy": []
        },
        "BNN w Meta": {
            "model": models.BNN([input_size, 4096, 4096, 10], init='uniform', std=STD, device=DEVICE),
            "parameters": {'n_epochs': N_EPOCHS, 'learning_rate': LEARNING_RATE,
                           'weight_decay': WEIGHT_DECAY, 'metaplasticity': METAPLASTICITY, **all_test},
            "accuracy": []
        }
    }

    for name, data in networks_data.items():
        full_name = os.path.join(SAVE_FOLDER, name)
        folder = versionning(full_name, name)
        os.makedirs(folder, exist_ok=True)
        print(f"Training {name}...")
        for train_dataset in [mnist_train, fashion_mnist_train]:
            data['accuracy'].append(data['model'].train_network(
                train_dataset, **data['parameters']))

        print(f"Saving {name} weights, accuracy and figure...")
        weights_name = name + "-weights"
        data['model'].save_state(versionning(
            folder, weights_name, ".pt"))

        print(f"Saving {name} accuracy...")
        accuracy_name = name + "-accuracy"
        accuracy = torch.tensor(data['accuracy'][0] + data['accuracy'][1])
        torch.save(accuracy, versionning(
            folder, accuracy_name, ".pt"))

        print(f"Exporting visualisation of {name} accuracy...")
        title = name + "-MNIST-FashionMNIST-accuracy"
        visualize_sequential(title, accuracy, folder=folder)

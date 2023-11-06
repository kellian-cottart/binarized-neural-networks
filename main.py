from utils import *
import models
import torch
import matplotlib.pyplot as plt
import os
import datetime

### GLOBAL VARIABLES ###
SEED = 2506
BATCH_SIZE = 100
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-8
METAPLASTICITY = 1.5
N_EPOCHS = 50
STD = 0.1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### PATHS ###
DATASETS_PATH = "./datasets"
ARRAY_PATH = "./saved_arrays"
WEIGHTS_PATH = "./saved_weights"
FIGURE_PATH = "./figures"


def versionning(folder, title, format=".pdf"):
    os.makedirs(folder, exist_ok=True)
    # YYYY-MM-DD-title-version
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    version = 1
    while os.path.exists(os.path.join(folder, f"{timestamp}-{title}-v{version}")):
        version += 1
    versionned = os.path.join(folder, f"{timestamp}-{title}-v{version}")
    return versionned + format


def visualize_sequential(title, t_accuracy):
    plt.figure()
    plt.plot(range(1, len(t_accuracy)+1), t_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['MNIST', 'Fashion-MNIST'])
    os.makedirs(FIGURE_PATH, exist_ok=True)
    versionned = versionning(FIGURE_PATH, title)
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
            "train": [mnist_train, fashion_mnist_train],
            "model": models.DNN([input_size, 4096, 4096, 10], init='uniform', std=STD, device=DEVICE),
            "parameters": {'n_epochs': N_EPOCHS, 'learning_rate': LEARNING_RATE,
                           'weight_decay': WEIGHT_DECAY, **all_test},
            "accuracy": []
        },
        "BNN w/out Meta": {
            "train": [mnist_train, fashion_mnist_train],
            "model": models.BNN([input_size, 4096, 4096, 10], init='uniform', std=STD, device=DEVICE),
            "parameters": {'n_epochs': N_EPOCHS, 'learning_rate': LEARNING_RATE,
                           'weight_decay': WEIGHT_DECAY, 'metaplasticity': 0, **all_test},
            "accuracy": []
        },
        "BNN w/ Meta": {
            "train": [mnist_train, fashion_mnist_train],
            "model": models.BNN([input_size, 4096, 4096, 10], init='uniform', std=STD, device=DEVICE),
            "parameters": {'n_epochs': N_EPOCHS, 'learning_rate': LEARNING_RATE,
                           'weight_decay': WEIGHT_DECAY, 'metaplasticity': METAPLASTICITY, **all_test},
            "accuracy": []
        }
    }

    for name, data in networks_data.items():
        print(f"Training {name}...")
        for train_dataset in data['train']:
            data['accuracy'].append(data['model'].train_network(
                train_dataset, **data['parameters']))

        print(f"Saving {name} weights, accuracy and figure...")
        weights_name = name + "-weights"
        data['model'].save_weights(versionning(
            WEIGHTS_PATH, weights_name, ".pt"))

        print(f"Saving {name} accuracy...")
        accuracy_name = name + "-accuracy"
        accuracy = torch.tensor(data['accuracy'])
        torch.save(accuracy, versionning(
            ARRAY_PATH, accuracy_name, ".pt"))

        print(f"Exporting visualisation of {name} accuracy...")
        title = name + "-MNIST-FashionMNIST-accuracy"
        visualize_sequential(title, accuracy)

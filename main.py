from utils import *
import models
import torch
import matplotlib.pyplot as plt
import os

### GLOBAL VARIABLES ###
SEED = 2506
DATASETS_PATH = "./datasets"
ARRAY_PATH = "./saved_arrays"
FIGURE_PATH = "./figures"
BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize():
    # load t_accuracy
    t_accuracy = torch.load(os.path.join(ARRAY_PATH, 't_accuracy.pt'))
    plt.figure()
    plt.plot(range(1, len(t_accuracy)+1), t_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['MNIST', 'Fashion-MNIST'])
    os.makedirs(FIGURE_PATH, exist_ok=True)
    plt.savefig(os.path.join(
        FIGURE_PATH, '2023-11-03-dnn-mnist-vs-fmnist.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    ### SEED ###
    torch.manual_seed(SEED)

    ### LOAD DATASETS ###
    mnist_train, mnist_test = mnist(DATASETS_PATH, BATCH_SIZE)
    fashion_mnist_train, fashion_mnist_test = fashion_mnist(
        DATASETS_PATH, BATCH_SIZE)

    all_test = {'mnist_test': mnist_test,
                'fashion_mnist_test': fashion_mnist_test}

    ### CREATE DNN ###
    input_size = mnist_train.dataset.data.shape[1] * \
        mnist_train.dataset.data.shape[2]
    dnn = models.DNN([input_size, 128, 64, 10], init='gauss', device=DEVICE)

    ### TRAIN DNN w/ MNIST ###
    mnist_accuracy = dnn.train_network(
        mnist_train, n_epochs=50, learning_rate=0.01, **all_test)

    ### TRAIN DNN w/ Fashion-MNIST ###
    fmnist_accuracy = dnn.train_network(fashion_mnist_train, n_epochs=50,
                                        learning_rate=0.01, **all_test)

    ### VISUALIZE ACCURACY ###
    t_epoch = 100
    t_accuracy = torch.tensor(mnist_accuracy + fmnist_accuracy)
    os.makedirs(ARRAY_PATH, exist_ok=True)
    torch.save(t_accuracy, os.path.join(ARRAY_PATH, 't_accuracy.pt'))

from torchvision import datasets
import os
PATH_MNIST_X_TRAIN = "datasets/MNIST/raw/train-images-idx3-ubyte"
PATH_MNIST_Y_TRAIN = "datasets/MNIST/raw/train-labels-idx1-ubyte"
PATH_MNIST_X_TEST = "datasets/MNIST/raw/t10k-images-idx3-ubyte"
PATH_MNIST_Y_TEST = "datasets/MNIST/raw/t10k-labels-idx1-ubyte"

PATH_FASHION_MNIST_X_TRAIN = "datasets/FashionMNIST/raw/train-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TRAIN = "datasets/FashionMNIST/raw/train-labels-idx1-ubyte"
PATH_FASHION_MNIST_X_TEST = "datasets/FashionMNIST/raw/t10k-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TEST = "datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte"


def download_mnist():
    datasets.MNIST("datasets", download=True)


def download_fashion_mnist():
    datasets.FashionMNIST("datasets", download=True)


def mnist(loader, batch_size, permute_idx=None):
    if not os.path.exists(PATH_MNIST_X_TRAIN):
        download_mnist()

    mnist_train, mnist_test = loader(
        batch_size=batch_size,
        path_train_x=PATH_MNIST_X_TRAIN,
        path_train_y=PATH_MNIST_Y_TRAIN,
        path_test_x=PATH_MNIST_X_TEST,
        path_test_y=PATH_MNIST_Y_TEST,
        permute_idx=permute_idx,
    )
    return mnist_train, mnist_test


def fashion_mnist(loader, batch_size):
    if not os.path.exists(PATH_FASHION_MNIST_X_TRAIN):
        download_fashion_mnist()

    fashion_mnist_train, fashion_mnist_test = loader(
        batch_size=batch_size,
        path_train_x=PATH_FASHION_MNIST_X_TRAIN,
        path_train_y=PATH_FASHION_MNIST_Y_TRAIN,
        path_test_x=PATH_FASHION_MNIST_X_TEST,
        path_test_y=PATH_FASHION_MNIST_Y_TEST,
    )
    return fashion_mnist_train, fashion_mnist_test

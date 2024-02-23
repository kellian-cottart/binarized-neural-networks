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

PATH_CIFAR10 = "datasets/cifar-10-batches-py"
PATH_DATABATCH = [f"{PATH_CIFAR10}/data_batch_{i}" for i in range(1, 6)]
PATH_TESTBATCH = f"{PATH_CIFAR10}/test_batch"


def download_mnist():
    datasets.MNIST("datasets", download=True)


def download_fashion_mnist():
    datasets.FashionMNIST("datasets", download=True)


def download_cifar10():
    datasets.CIFAR10("datasets", download=True)


def mnist(loader, batch_size, permute_idx=None):
    if not os.path.exists(PATH_MNIST_X_TRAIN):
        download_mnist()

    mnist_train, mnist_test = loader.mnist(
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

    fashion_mnist_train, fashion_mnist_test = loader.mnist(
        batch_size=batch_size,
        path_train_x=PATH_FASHION_MNIST_X_TRAIN,
        path_train_y=PATH_FASHION_MNIST_Y_TRAIN,
        path_test_x=PATH_FASHION_MNIST_X_TEST,
        path_test_y=PATH_FASHION_MNIST_Y_TEST,
    )
    return fashion_mnist_train, fashion_mnist_test


def cifar10(loader, batch_size):
    if not os.path.exists("datasets/CIFAR10/raw"):
        download_cifar10()
    cifar10_train, cifar10_test = loader.cifar10(
        batch_size=batch_size,
        path_databatch=PATH_DATABATCH,
        path_testbatch=PATH_TESTBATCH,
    )
    return cifar10_train, cifar10_test

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
PATH_CIFAR10_DATABATCH = [
    f"{PATH_CIFAR10}/data_batch_{i}" for i in range(1, 6)]
PATH_CIFAR10_TESTBATCH = f"{PATH_CIFAR10}/test_batch"

PATH_CIFAR100 = "datasets/cifar-100-python"
PATH_CIFAR100_DATABATCH = [f"{PATH_CIFAR100}/train"]
PATH_CIFAR100_TESTBATCH = f"{PATH_CIFAR100}/test"


def mnist(loader, batch_size, permute_idx=None):
    if not os.path.exists(PATH_MNIST_X_TRAIN):
        datasets.MNIST("datasets", download=True)

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
        datasets.FashionMNIST("datasets", download=True)

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
        datasets.CIFAR10("datasets", download=True)
    cifar10_train, cifar10_test = loader.cifar10(
        batch_size=batch_size,
        path_databatch=PATH_CIFAR10_DATABATCH,
        path_testbatch=PATH_CIFAR10_TESTBATCH,
    )
    return cifar10_train, cifar10_test


def cifar100(loader, batch_size):
    if not os.path.exists("datasets/CIFAR100/raw"):
        datasets.CIFAR100("datasets", download=True)
    cifar100_train, cifar100_test = loader.cifar100(
        batch_size=batch_size,
        path_databatch=PATH_CIFAR100_DATABATCH,
        path_testbatch=PATH_CIFAR100_TESTBATCH,
    )
    return cifar100_train, cifar100_test

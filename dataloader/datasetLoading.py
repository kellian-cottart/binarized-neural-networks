from torchvision import datasets
import os
import torch

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


def task_selection(loader, task, batch_size, *args, **kwargs):
    """ Select the task to load

    Args:
        task (str): Name of the task
        batch_size (int): Batch size
        shape (tuple): Shape of the input

    """
    ### INIT DATASET ###
    if task == "Sequential":
        mnist_train, mnist_test = mnist(loader, batch_size)
        fashion_train, fashion_test = fashion_mnist(
            loader, batch_size)
        train_loader = [mnist_train, fashion_train]
        test_loader = [mnist_test, fashion_test]
        shape = mnist_train.dataset[0][0].shape
        target_size = len(mnist_train.dataset.targets.unique())
    elif task == "PermutedMNIST" or task == "MNIST":
        # load mnist
        mnist_train, mnist_test = mnist(loader, batch_size)
        shape = mnist_train.dataset[0][0].shape
        target_size = len(mnist_train.dataset.targets.unique())
        train_loader = [mnist_train]
        test_loader = [mnist_test]
    elif task == "CIFAR10":
        cifar10_train, cifar10_test = cifar10(
            loader, batch_size=batch_size)
        shape = cifar10_train.dataset[0][0].shape
        target_size = len(cifar10_train.dataset.targets.unique())
        train_loader = [cifar10_train]
        test_loader = [cifar10_test]
    elif task == "CIFAR100" or task == "CIFAR100INCREMENTAL":
        cifar100_train, cifar100_test = cifar100(
            loader, batch_size=batch_size)
        shape = cifar100_train.dataset[0][0].shape
        target_size = len(cifar100_train.dataset.targets.unique())
        train_loader = [cifar100_train]
        test_loader = [cifar100_test]
    else:
        raise ValueError(
            f"Task {task} is not implemented.")

    # if there are less than 4 elements in shape, add channels as 1
    if len(shape) < 3:
        shape = (1, *shape)

    return train_loader, test_loader, shape, target_size

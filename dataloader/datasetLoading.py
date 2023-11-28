from torchvision import datasets
from .cpuLoading import CPULoading

PATH_MNIST_X_TRAIN = "datasets/MNIST/raw/train-images-idx3-ubyte"
PATH_MNIST_Y_TRAIN = "datasets/MNIST/raw/train-labels-idx1-ubyte"
PATH_MNIST_X_TEST = "datasets/MNIST/raw/t10k-images-idx3-ubyte"
PATH_MNIST_Y_TEST = "datasets/MNIST/raw/t10k-labels-idx1-ubyte"

PATH_FASHION_MNIST_X_TRAIN = "datasets/FashionMNIST/raw/train-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TRAIN = "datasets/FashionMNIST/raw/train-labels-idx1-ubyte"
PATH_FASHION_MNIST_X_TEST = "datasets/FashionMNIST/raw/t10k-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TEST = "datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte"


def mnist(loader, permute_idx=None, as_dataset=True):

    if isinstance(loader, CPULoading):
        mnist_train, mnist_test = loader(datasets.MNIST)
    else:
        if as_dataset:
            mnist_train, mnist_test = loader(
                path_train_x=PATH_MNIST_X_TRAIN,
                path_train_y=PATH_MNIST_Y_TRAIN,
                path_test_x=PATH_MNIST_X_TEST,
                path_test_y=PATH_MNIST_Y_TEST,
                permute_idx=permute_idx,
            )
        else:
            mnist_train, mnist_test = loader.simple_dataset(
                path_train_x=PATH_MNIST_X_TRAIN,
                path_train_y=PATH_MNIST_Y_TRAIN,
                path_test_x=PATH_MNIST_X_TEST,
                path_test_y=PATH_MNIST_Y_TEST,
                permute_idx=permute_idx,
            )
    return mnist_train, mnist_test


def fashion_mnist(loader, as_dataset=True):
    if isinstance(loader, CPULoading):
        fashion_mnist_train, fashion_mnist_test = loader(datasets.FashionMNIST)
    else:
        if as_dataset:
            fashion_mnist_train, fashion_mnist_test = loader(
                path_train_x=PATH_FASHION_MNIST_X_TRAIN,
                path_train_y=PATH_FASHION_MNIST_Y_TRAIN,
                path_test_x=PATH_FASHION_MNIST_X_TEST,
                path_test_y=PATH_FASHION_MNIST_Y_TEST,
            )
        else:
            fashion_mnist_train, fashion_mnist_test = loader.simple_dataset(
                path_train_x=PATH_FASHION_MNIST_X_TRAIN,
                path_train_y=PATH_FASHION_MNIST_Y_TRAIN,
                path_test_x=PATH_FASHION_MNIST_X_TEST,
                path_test_y=PATH_FASHION_MNIST_Y_TEST,
            )
    return fashion_mnist_train, fashion_mnist_test

from torchvision import datasets
from .cpuLoading import CPULoading


def mnist(loader, permute_idx=None):
    if isinstance(loader, CPULoading):
        mnist_train, mnist_test = loader(
            datasets.MNIST, permute_idx=permute_idx)
    else:
        mnist_train, mnist_test = loader(
            path_train_x="datasets/MNIST/raw/train-images-idx3-ubyte",
            path_train_y="datasets/MNIST/raw/train-labels-idx1-ubyte",
            path_test_x="datasets/MNIST/raw/t10k-images-idx3-ubyte",
            path_test_y="datasets/MNIST/raw/t10k-labels-idx1-ubyte",
            permute_idx=permute_idx,
        )
    return mnist_train, mnist_test


def fashion_mnist(loader):
    if isinstance(loader, CPULoading):
        fashion_mnist_train, fashion_mnist_test = loader(datasets.FashionMNIST)
    else:

        fashion_mnist_train, fashion_mnist_test = loader(
            path_train_x="datasets/FashionMNIST/raw/train-images-idx3-ubyte",
            path_train_y="datasets/FashionMNIST/raw/train-labels-idx1-ubyte",
            path_test_x="datasets/FashionMNIST/raw/t10k-images-idx3-ubyte",
            path_test_y="datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte",
        )
    return fashion_mnist_train, fashion_mnist_test

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import PIL.Image as Image
import torch
# change get item in mnist to convert first to cpu then to cuda


def mnist(path, batch_size, num_workers=0):
    """ Load MNIST dataset

    Args:
        path (str): Path to save/load dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading

    Returns:
        torch.utils.data.DataLoader: MNIST training dataset
        torch.utils.data.DataLoader: MNIST testing dataset
    """
    ### NORMALIZATION ###
    mean, std = 0.1307, 0.3081  # As in EWC paper
    normalisation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ### DOWNLOAD DATASETS ###
    mnist_train = datasets.MNIST(
        root=path, download=True, transform=normalisation, target_transform=None, train=True)
    mnist_test = datasets.MNIST(
        root=path, download=True, transform=normalisation, target_transform=None, train=False)

    ### DATA LOADER ###
    mnist_train = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mnist_test = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return mnist_train, mnist_test

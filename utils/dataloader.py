from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def mnist(path, batch_size):
    """ Load MNIST dataset

    Args:
        path (str): Path to save/load dataset
        batch_size (int): Batch size for training

    Returns:
        torch.utils.data.DataLoader: MNIST training dataset
        torch.utils.data.DataLoader: MNIST testing dataset
    """
    ### NORMALIZATION ###
    mean, std = 0, 1
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
        mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False)

    return mnist_train, mnist_test

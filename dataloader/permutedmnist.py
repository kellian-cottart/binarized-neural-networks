from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import random


class PermutedMNIST(datasets.MNIST):
    """ Permuted MNIST dataset

    Extension of the MNIST dataset where the pixels are permuted according to a random permutation
    Used for the Continual Learning experiments
    """

    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28
        # Set the permutation
        self.data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                 for img in self.data])

    def __getitem__(self, index):
        """ Get an item from the dataset

        Args:
            index (int): Index of the sample to return
        """
        # Return the image and the label
        img, target = self.data[index], self.targets[index]
        return img, target

    def get_sample(self, sample_size):
        """ Get a sample of the dataset

        Args:
            sample_size (int): Number of samples to return
        """
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]


def permuted_mnist(path, batch_size):
    """ Load Permuted MNIST dataset

    Args:
        path (str): Path to save/load dataset
        batch_size (int): Batch size for training

    Returns:
        torch.utils.data.DataLoader: Permuted MNIST training dataset
        torch.utils.data.DataLoader: Permuted MNIST testing dataset
    """
    ### NORMALIZATION ###
    mean, std = 0.1307, 0.3081  # As in EWC paper
    normalisation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ### DOWNLOAD DATASETS ###
    permute_idx = torch.randperm(28 * 28)
    permuted_mnist_train = PermutedMNIST(
        root=path, train=True, permute_idx=permute_idx)
    permuted_mnist_test = PermutedMNIST(
        root=path, train=False, permute_idx=permute_idx)

    ### DATA LOADER ###
    permuted_mnist_train = DataLoader(
        permuted_mnist_train, batch_size=batch_size, shuffle=True)
    permuted_mnist_test = DataLoader(
        permuted_mnist_test, batch_size=batch_size, shuffle=False)

    return permuted_mnist_train, permuted_mnist_test

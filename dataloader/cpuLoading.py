from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import random
import numpy as np


class CPULoading:
    """ Load torchvision datasets on CPU

    Args:
        path (str): Path to the datasets
        batch_size (int): Batch size
        mean (float, optional): Mean of the dataset. Defaults to 0.1307.
        std (float, optional): Standard deviation of the dataset. Defaults to 0.3081.
        padding (int, optional): Padding to add to the images. Defaults to 0.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
    """

    def __init__(self, path, batch_size, mean=0.1307, std=0.3081, padding=0, num_workers=0, *args, **kwargs):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.padding = padding
        self.normalisation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, dataset, *args, **kwargs):
        """ Load any dataset

        Args:
            dataset (torchvision.datasets): Dataset to load

        Returns:
            torch.utils.data.DataLoader: Training dataset
            torch.utils.data.DataLoader: Testing dataset
        """
        train = dataset(
            root=self.path, download=True, transform=self.normalisation, target_transform=None, train=True, *args, **kwargs)
        test = dataset(
            root=self.path, download=True, transform=self.normalisation, target_transform=None, train=False, *args, **kwargs)

        # add padding
        padding = self.padding
        train.data = torch.from_numpy(np.pad(train.data, ((0, 0), (padding, padding),
                                                          (padding, padding)), 'constant'))
        test.data = torch.from_numpy(np.pad(test.data, ((0, 0), (padding, padding),
                                                        (padding, padding)), 'constant'))

        train = DataLoader(
            train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test = DataLoader(
            test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train, test


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

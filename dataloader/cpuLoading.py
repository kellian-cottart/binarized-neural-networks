from torchvision import transforms
from torch.utils.data import DataLoader
import torch


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

    def __init__(self, path, padding=0, num_workers=0, *args, **kwargs):
        self.path = path
        self.num_workers = num_workers
        self.padding = padding

    def __call__(self, batch_size, dataset, permute_idx=None, *args, **kwargs):
        """ Load any dataset

        Args:
            batch_size (int): Batch size    
            dataset (torchvision.datasets): Dataset to load
            permute_idx (list, optional): Permutation of the pixels. Defaults to None.

        Returns:
            torch.utils.data.DataLoader: Training dataset
            torch.utils.data.DataLoader: Testing dataset
        """
        normalisation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x/255),
            transforms.Normalize((0,), (1,)),
        ])

        train = dataset(root=self.path,
                        train=True,
                        download=True,
                        target_transform=None,
                        transform=normalisation)
        test = dataset(root=self.path,
                       train=False,
                       download=True,
                       target_transform=None,
                       transform=normalisation)

        current_size = train.data.shape[1]

        target_size = (current_size+self.padding*2)

        # if permute_idx is given, permute the dataset as for PermutedMNIST
        if "permute_idx" in kwargs and kwargs["permute_idx"] is not None:
            # Flatten the images
            train.data = train.data.reshape(train.data.shape[0], -1)
            test.data = test.data.reshape(test.data.shape[0], -1)
            # Add padding (Ref: Chen Zeno - Task Agnostic Continual Learning Using Online Variational Bayes)
            train.data = torch.cat(
                (train.data, torch.zeros(len(train.data), target_size**2-current_size**2)), axis=1)
            test.data = torch.cat(
                (test.data, torch.zeros(len(test.data), target_size**2-current_size**2)), axis=1)
            # permute_idx is the permutation to apply to the pixels of the images
            permute_idx = kwargs["permute_idx"]
            # Permute the pixels of the training examples using torch
            train.data = train.data[:, permute_idx]
            # Permute the pixels of the test examples
            test.data = test.data[:, permute_idx]
        else:
            # regular padding
            train.data = torch.nn.functional.pad(
                train.data, (self.padding, self.padding, self.padding, self.padding))
            test.data = torch.nn.functional.pad(
                test.data, (self.padding, self.padding, self.padding, self.padding))
            # flatten the images
            train.data = train.data.reshape(train.data.shape[0], -1)
            test.data = test.data.reshape(test.data.shape[0], -1)

        train = DataLoader(train, batch_size=batch_size,
                           shuffle=True, num_workers=self.num_workers)
        max_batch_size = len(test)
        test = DataLoader(test, batch_size=max_batch_size,
                          shuffle=False, num_workers=self.num_workers)
        return train, test

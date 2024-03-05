import torch
import numpy as np
import idx2numpy
from torchvision.transforms import v2
import pickle


class GPUTensorDataset(torch.utils.data.Dataset):
    """ TensorDataset which has a data and a targets tensor and allows batching

    Args:
        data (torch.tensor): Data tensor
        targets (torch.tensor): Targets tensor
        device (str, optional): Device to use. Defaults to "cuda:0".
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.  
    """

    def __init__(self, data, targets, device="cuda:0", transform=None):
        self.data = data.to(device)
        self.targets = targets.to(device)
        self.transform = transform

    def __getitem__(self, index):
        """ Return a (data, target) pair """
        if self.transform:
            return self.transform(self.data[index]), self.targets[index]
        return self.data[index], self.targets[index]

    def __len__(self):
        """ Return the number of samples """
        return len(self.data)


class GPUDataLoader():
    """ DataLoader which has a data and a targets tensor and allows batching

    Args:
        dataset (GPUTensorDataset): Dataset to load
        batch_size (int): Batch size
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it is not full. Defaults to True.
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        """ Return an iterator over the dataset """
        self.index = 0
        self.perm = torch.randperm(len(self.dataset)) if self.shuffle else torch.arange(
            len(self.dataset))
        return self

    def __next__(self):
        """ Return a (data, target) pair """
        if self.index >= len(self.dataset):
            raise StopIteration
        data = self.dataset.data[self.perm[self.index:self.index +
                                           self.batch_size]].clone()
        targets = self.dataset.targets[self.perm[self.index:self.index +
                                                 self.batch_size]].clone()
        self.index += self.batch_size
        return data, targets

    def __len__(self):
        """ Return the number of batches """
        return len(self.dataset)//self.batch_size

    def permute_dataset(self, permutations):
        """ Yield a list of DataLoaders with permuted pixels of the current dataset
        As much memory efficient as possible

        Args:
            permutations (list): List of permutations

        Returns:
            list: List of DataLoader with permuted pixels
        """
        # Save the reference data
        self.reference_data = self.dataset.data.detach().clone()
        # For each permutation, permute the pixels and yield the DataLoader
        for perm in permutations:
            self.dataset.data = self.reference_data.view(
                self.dataset.data.shape[0], -1)[:, perm].view(self.reference_data.shape)
            yield self
        # Reset the dataset
        self.dataset.data = self.reference_data
        # remove the reference data and targets
        del self.reference_data

    def class_incremental_dataset(self,
                                  n_tasks: int = 5,
                                  n_classes: int = 10,
                                  test: bool = False):
        """ Yield a list of DataLoaders with data only from the t // n_tasks task of the current dataset
        Then, yields the DataLoader with the whole dataset

        Args:
            n_tasks (int): Number of tasks to split the dataset into
            n_classes (int): Number of classes in the dataset to take per task
            test (bool, optional): If True, works with the test set and adjusts batch size. Defaults to False.
        """
        # Save the reference data and targets
        self.reference_data = self.dataset.data.detach().clone()
        self.reference_targets = self.dataset.targets.detach().clone()
        for i in range(n_tasks):
            # For each task, retrieve the indexes of the n_classes classes
            indexes = torch.nonzero(
                (self.reference_targets >= i * n_classes) & (self.reference_targets < (i + 1) * n_classes)).squeeze()
            self.dataset.data = self.reference_data[indexes].detach().clone()
            self.dataset.targets = self.reference_targets[indexes].detach(
            ).clone()
            if test:
                self.batch_size = len(self.dataset)
            yield self
        # Reset the dataset
        self.dataset.data = self.reference_data
        self.dataset.targets = self.reference_targets
        if test:
            self.batch_size = len(self.dataset)
        # remove the reference data and targets
        del self.reference_data
        del self.reference_targets


class GPULoading:
    """ Load local datasets on GPU

    Args:
        batch_size (int): Batch size
        padding (int, optional): Padding to add to the images. Defaults to 0.
        as_dataset (bool, optional): If True, returns a TensorDataset, else returns a DataLoader. Defaults to False.
    """

    def __init__(self, padding=0, device="cuda:0", as_dataset=False, *args, **kwargs):
        self.padding = padding
        self.device = device
        self.as_dataset = as_dataset

    def batching(self, train_x, train_y, test_x, test_y, batch_size, data_augmentation=False):
        """ Create a DataLoader to load the data in batches

        Args:
            train_x (torch.tensor): Training data
            train_y (torch.tensor): Training labels
            test_x (torch.tensor): Testing data
            test_y (torch.tensor): Testing labels
            batch_size (int): Batch size
            data_augmentation (bool, optional): Whether to use data augmentation. Defaults to False.

        Returns:
            DataLoader, DataLoader: Training and testing DataLoader

        """
        # Data augmentation
        transform = v2.Compose([
            v2.RandomChoice([
                v2.RandomResizedCrop(
                    size=(train_x.shape[-1], train_x.shape[-1]),
                    antialias=True,
                    scale=(0.75, 1),
                ),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ]),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=(-15, 15), translate=(0, 0.1)),
        ])

        # Converting the data to a GPU TensorDataset (allows to load everything in the GPU memory at once)
        train_dataset = GPUTensorDataset(
            train_x, torch.Tensor(train_y).type(
                torch.LongTensor), device=self.device, transform=transform if data_augmentation else None)
        test_dataset = GPUTensorDataset(test_x.float(), torch.Tensor(test_y).type(
            torch.LongTensor), device=self.device)
        max_batch_size = len(test_dataset)

        if not self.as_dataset:
            # create a DataLoader to load the data in batches
            train_dataset = GPUDataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_dataset = GPUDataLoader(
                test_dataset, batch_size=max_batch_size, shuffle=True)
        else:
            # create a DataLoader to load the data in batches
            train_dataset = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_dataset = torch.utils.data.DataLoader(
                test_dataset, batch_size=max_batch_size, shuffle=True)
        return train_dataset, test_dataset

    def pad(self, train_x, test_x):
        """ Add padding to the images if necessary"""
        # Add padding if necessary
        train_x = torch.nn.functional.pad(
            train_x, (self.padding, self.padding, self.padding, self.padding))
        test_x = torch.nn.functional.pad(
            test_x, (self.padding, self.padding, self.padding, self.padding))
        return train_x, test_x

    def normalization(self, train_x, test_x):
        """ Normalize the pixels in train_x and test_x using transform

        Args:
            train_x (np.array): Training data
            test_x (np.array): Testing data

        Returns:
            torch.tensor, torch.tensor: Normalized training and testing data
        """
        # Completely convert train_x and test_x to float torch tensors
        # division by 255 is only scaling from uint to float
        train_x = torch.from_numpy(train_x).float() / 255
        test_x = torch.from_numpy(test_x).float() / 255
        # Normalize the pixels
        transform = v2.Compose([v2.Normalize((0,), (1,))])
        return transform(train_x), transform(test_x)

    def mnist(self, batch_size, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
        """ Load a local dataset on GPU corresponding either to MNIST or FashionMNIST

        Args:
            batch_size (int): Batch size
            path_train_x (str): Path to the training data
            path_train_y (str): Path to the training labels
            path_test_x (str): Path to the testing data
            path_test_y (str): Path to the testing labels
        """
        # load ubyte dataset
        train_x = idx2numpy.convert_from_file(
            path_train_x).astype(np.float32)
        train_y = idx2numpy.convert_from_file(
            path_train_y).astype(np.float32)
        test_x = idx2numpy.convert_from_file(
            path_test_x).astype(np.float32)
        test_y = idx2numpy.convert_from_file(
            path_test_y).astype(np.float32)
        # Normalize and pad the data
        train_x, test_x = self.normalization(train_x, test_x)
        train_x, test_x = self.pad(train_x, test_x)
        return self.batching(train_x, train_y, test_x, test_y, batch_size)

    def cifar10(self, batch_size, path_databatch, path_testbatch, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR10 """
        # Deal with the training data
        train_x = []
        train_y = []
        for path in path_databatch:
            with open(path, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
            train_x.append(dict[b'data'])
            train_y.append(dict[b'labels'])
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)

        # Deal with the test data
        with open(path_testbatch, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        test_x = dict[b'data']
        test_y = dict[b'labels']

        # Deflatten the data
        train_x = train_x.reshape(-1, 3, 32, 32)
        test_x = test_x.reshape(-1, 3, 32, 32)

        # Normalize and pad the data
        train_x, test_x = self.normalization(train_x, test_x)
        train_x, test_x = self.pad(train_x, test_x)
        return self.batching(train_x, train_y, test_x, test_y, batch_size, data_augmentation=True)

    def cifar100(self, batch_size, path_databatch, path_testbatch):
        """ Load a local dataset on GPU corresponding to CIFAR100 """
        with open(path_databatch[0], "rb") as f:
            data = pickle.load(f, encoding="bytes")
            train_x = data[b"data"]
            train_y = data[b"fine_labels"]
        with open(path_testbatch, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            test_x = data[b"data"]
            test_y = data[b"fine_labels"]

        # Deflatten the data
        train_x = train_x.reshape(-1, 3, 32, 32)
        test_x = test_x.reshape(-1, 3, 32, 32)

        # Normalize and pad the data
        train_x, test_x = self.normalization(
            train_x, test_x)
        train_x, test_x = self.pad(train_x, test_x)
        return self.batching(train_x, train_y, test_x, test_y, batch_size, data_augmentation=True)

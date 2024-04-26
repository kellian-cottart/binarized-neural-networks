import torch
import numpy as np
import idx2numpy
from torchvision.transforms import v2
from torchvision import models, datasets
import pickle
import os
import time


class GPUTensorDataset(torch.utils.data.Dataset):
    """ TensorDataset which has a data and a targets tensor and allows batching

    Args:
        data (torch.tensor): Data tensor
        targets (torch.tensor): Targets tensor
        device (str, optional): Device to use. Defaults to "cuda:0".
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.
    """

    def __init__(self, data, targets, device="cuda:0"):
        self.data = data.to(device)
        self.targets = targets.to(device)

    def __getitem__(self, index):
        """ Return a (data, target) pair """
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

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, transform=None, device="cuda:0", test=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transform = transform
        self.device = device
        self.test = test

    def __iter__(self):
        """ Return an iterator over the dataset """
        self.index = 0
        if self.shuffle:
            self.perm = torch.randperm(len(self.dataset))
        else:
            self.perm = torch.arange(len(self.dataset))
        return self

    def __next__(self):
        """ Return a (data, target) pair """
        if self.index >= len(self.dataset):
            raise StopIteration
        if self.index + self.batch_size > len(self.dataset):
            if self.drop_last:
                raise StopIteration
            else:
                indexes = self.perm[self.index:]
        else:
            indexes = self.perm[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        data, targets = self.dataset[indexes]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        """ Return the number of batches """
        return len(self.dataset)//self.batch_size

    def permute_dataset(self, permutations):
        """ Yield DataLoaders with permuted pixels of the current dataset

        Args:
            permutations (list): List of permutations

        Returns:
            list: List of DataLoader with permuted pixels
        """
        # For each permutation, permute the pixels and yield the DataLoader
        start = time.time()
        for perm in permutations:
            # Create a new GPUDataset
            dataset = GPUTensorDataset(
                self.dataset.data.view(
                    self.dataset.data.shape[0], -1)[:, perm].view(self.dataset.data.shape), self.dataset.targets, device=self.device)
            if not self.test:
                yield GPUDataLoader(dataset,
                                    self.batch_size,
                                    self.shuffle,
                                    self.drop_last,
                                    self.transform,
                                    self.device)
            else:
                yield GPUDataLoader(dataset,
                                    dataset.data.size(0),
                                    self.shuffle,
                                    False,
                                    None,
                                    self.device)

    def class_incremental_dataset(self, permutations):
        """ Yield DataLoaders with data only from the classes in permutations

        Args:
            permutations (list): List of permutations of the classes
        Yields:
            GPUDataLoader: DataLoader
        """
        for perm in permutations:
            indexes = torch.isin(self.dataset.targets, perm).nonzero(
                as_tuple=False).squeeze()
            # Create a new GPUDataset
            dataset = GPUTensorDataset(
                self.dataset.data[indexes], self.dataset.targets[indexes], device=self.device)
            if not self.test:
                yield GPUDataLoader(dataset,
                                    self.batch_size,
                                    self.shuffle,
                                    self.drop_last,
                                    self.transform,
                                    self.device)
            else:
                yield GPUDataLoader(dataset,
                                    dataset.data.size(0),
                                    self.shuffle,
                                    False,
                                    None,
                                    self.device)

    def stream_dataset(self, n_subsets):
        """ Yield DataLoaders with data from the dataset split into n_subsets subsets.

        Args:
            n_subsets (int): Number of subsets

        Yields:
            GPUDataLoader: DataLoader with data from the dataset split into n_subsets subsets

        """
        # Split the data into n_subsets with label and data
        for i in range(n_subsets):
            dataset = GPUTensorDataset(
                self.dataset.data[i::n_subsets], self.dataset.targets[i::n_subsets], device=self.device)
            if not self.test:
                yield GPUDataLoader(dataset,
                                    self.batch_size,
                                    self.shuffle,
                                    self.drop_last,
                                    self.transform,
                                    self.device)
            else:
                yield GPUDataLoader(dataset,
                                    dataset.data.size(0),
                                    self.shuffle,
                                    False,
                                    None,
                                    self.device)


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
            v2.RandomHorizontalFlip(),
            v2.Normalize((0,), (1,))
        ])

        # Converting the data to a GPU TensorDataset (allows to load everything in the GPU memory at once)
        train_dataset = GPUTensorDataset(
            train_x, torch.Tensor(train_y).type(
                torch.LongTensor), device=self.device if not data_augmentation else "cpu")
        test_dataset = GPUTensorDataset(test_x, torch.Tensor(test_y).type(
            torch.LongTensor), device=self.device if not data_augmentation else "cpu")

        max_batch_size = test_x.size(
            0) if train_dataset.data.device != "cpu" else batch_size

        if not self.as_dataset:
            # create a DataLoader to load the data in batches
            train_dataset = GPUDataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, transform=transform if data_augmentation else None, device=self.device)
            test_dataset = GPUDataLoader(
                test_dataset, batch_size=max_batch_size, shuffle=False, device=self.device, test=True)
        else:
            # create a DataLoader to load the data in batches
            train_dataset = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_dataset = torch.utils.data.DataLoader(
                test_dataset, batch_size=max_batch_size, shuffle=False)
        return train_dataset, test_dataset

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

        if len(train_x.size()) == 3:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)

        # Normalize the pixels to 0, 1
        transform = v2.Compose(
            [v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.Normalize((0,), (1,), inplace=True),
             v2.Pad(self.padding, fill=0, padding_mode='constant'),
             ])

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
        if "resize" in kwargs and kwargs["resize"] == True:
            folder = "datasets/cifar10_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder):
                self.feature_extraction(
                    folder, train_x, train_y, test_x, test_y, task="cifar10")
            train_x = torch.load(f"{folder}/cifar10_features_train.pt")
            train_y = torch.load(f"{folder}/cifar10_target_train.pt")
            test_x = torch.load(f"{folder}/cifar10_features_test.pt")
            test_y = torch.load(f"{folder}/cifar10_target_test.pt")
            # Normalize and pad the data
            return self.batching(train_x, train_y, test_x, test_y, batch_size)
        # Normalize and pad the data
        train_x, test_x = self.normalization(train_x, test_x)
        return self.batching(train_x, train_y, test_x, test_y, batch_size, data_augmentation=True)

    def cifar100(self, batch_size, path_databatch, path_testbatch, iterations=10, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR100 """
        with open(path_databatch[0], "rb") as f:
            data = pickle.load(f, encoding="bytes")
            train_x = data[b"data"]
            train_y = data[b"fine_labels"]
        with open(path_testbatch, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            test_x = data[b"data"]
            test_y = data[b"fine_labels"]
        train_x = train_x.reshape(-1, 3, 32, 32)
        test_x = test_x.reshape(-1, 3, 32, 32)
        if "resize" in kwargs and kwargs["resize"] == True:
            folder = "datasets/cifar100_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder) or not os.path.exists(f"{folder}/cifar100_{iterations}_features_train.pt"):
                self.feature_extraction(
                    folder, train_x, train_y, test_x, test_y, task="cifar100", iterations=iterations)
            train_x = torch.load(
                f"{folder}/cifar100_{iterations}_features_train.pt")
            train_y = torch.load(
                f"{folder}/cifar100_{iterations}_target_train.pt")
            test_x = torch.load(
                f"{folder}/cifar100_{iterations}_features_test.pt")
            test_y = torch.load(
                f"{folder}/cifar100_{iterations}_target_test.pt")
            # Normalize and pad the data
            return self.batching(train_x, train_y, test_x, test_y, batch_size)
        else:
            # Normalize and pad the data
            train_x, test_x = self.normalization(train_x, test_x)
            return self.batching(train_x, train_y, test_x, test_y, batch_size, data_augmentation=True)

    @torch.jit.export
    def feature_extraction(self, folder, train_x, train_y, test_x, test_y, task="cifar100", iterations=10):
        # The idea here is to use the resnet18 as feature extractor
        # Then create a new dataset with the extracted features from CIFAR100
        print(f"Extracting features from {task}...")
        resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        # Remove the classification layer
        resnet18 = torch.nn.Sequential(
            *list(resnet18.children())[:-1])
        # Freeze the weights of the feature extractor
        for param in resnet18.parameters():
            param.requires_grad = False
        # Transforms to apply to augment the data
        transform_train = v2.Compose([
            v2.Resize(220, antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.0,), std=(1.0,))
        ])
        transform_test = v2.Compose([
            v2.Resize(220, antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.0,), std=(1.0,))
        ])
        # Extract the features
        features_train = []
        target_train = []
        features_test = []
        target_test = []
        # Normalize
        train_x = torch.from_numpy(train_x).float() / 255
        test_x = torch.from_numpy(test_x).float() / 255
        if len(train_x.size()) == 3:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)
        # Converting the data to a GPU TensorDataset (allows to load everything in the GPU memory at once)
        train_dataset = GPUTensorDataset(
            train_x, torch.Tensor(train_y).type(
                torch.LongTensor), device=self.device)
        test_dataset = GPUTensorDataset(test_x, torch.Tensor(test_y).type(
            torch.LongTensor), device=self.device)
        train_dataset = GPUDataLoader(
            train_dataset, batch_size=1024, shuffle=True, drop_last=False, transform=transform_train, device=self.device)
        test_dataset = GPUDataLoader(
            test_dataset, batch_size=1024, shuffle=True, device=self.device, transform=transform_test)
        # Make n passes to extract the features
        for _ in range(iterations):
            for data, target in train_dataset:
                features_train.append(resnet18(data))
                target_train.append(target)
        for data, target in test_dataset:
            features_test.append(resnet18(data))
            target_test.append(target)

        # Concatenate the features
        features_train = torch.cat(features_train)
        target_train = torch.cat(target_train)
        features_test = torch.cat(features_test)
        target_test = torch.cat(target_test)
        # Save the features
        torch.save(features_train,
                   f"{folder}/{task}_{iterations}_features_train.pt")
        torch.save(
            target_train, f"{folder}/{task}_{iterations}_target_train.pt")
        torch.save(features_test,
                   f"{folder}/{task}_{iterations}_features_test.pt")
        torch.save(
            target_test, f"{folder}/{task}_{iterations}_target_test.pt")

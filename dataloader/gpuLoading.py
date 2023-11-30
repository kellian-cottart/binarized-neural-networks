import torch
import numpy as np
import idx2numpy
from torchvision import transforms


class GPUTensorDataset(torch.utils.data.Dataset):
    """ TensorDataset which has a data and a targets tensor and allows batching"""

    def __init__(self, data, targets, device="cuda:0"):
        self.data = data.to(device)
        self.targets = targets.to(torch.long).to(device)

    def __getitem__(self, index):
        """ Return a (data, target) pair """
        return self.data[index], self.targets[index]

    def __len__(self):
        """ Return the number of samples """
        return len(self.data)


class GPUDataLoader():
    """ DataLoader which has a data and a targets tensor and allows batching"""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

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
        batch = self.dataset[self.perm[self.index:self.index+self.batch_size]]
        self.index += self.batch_size
        return batch

    def __len__(self):
        """ Return the number of batches """
        return len(self.dataset) // self.batch_size


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

    def __call__(self, batch_size, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
        """ Load a local dataset

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

        current_size = train_x.shape[1]
        # Flatten the images
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)

        # Normalize the pixels in train_x and test_x using transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x/255.),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_x = transform(train_x).squeeze(0).to(
            self.device)
        test_x = transform(test_x).squeeze(0).to(
            self.device)

        # Add padding (Ref: Chen Zeno - Task Agnostic Continual Learning Using Online Variational Bayes)
        target_size = (current_size+self.padding*2)
        train_x = torch.cat(
            (train_x, torch.zeros(len(train_x), target_size**2-current_size**2).to(self.device)), axis=1)
        test_x = torch.cat(
            (test_x, torch.zeros(len(test_x), target_size**2-current_size**2).to(self.device)), axis=1)

        # if permute_idx is given, permute the dataset as for PermutedMNIST
        if "permute_idx" in kwargs and kwargs["permute_idx"] is not None:

            # permute_idx is the permutation to apply to the pixels of the images
            permute_idx = kwargs["permute_idx"]
            # Permute the pixels of the training examples using torch
            train_x = train_x[:, permute_idx]
            # Permute the pixels of the test examples
            test_x = test_x[:, permute_idx]

        train_dataset = GPUTensorDataset(
            train_x, torch.from_numpy(train_y))
        test_dataset = GPUTensorDataset(test_x, torch.from_numpy(test_y))
        max_batch_size = len(test_dataset)
        if not self.as_dataset:
            # create a DataLoader to load the data in batches
            train_dataset = GPUDataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = GPUDataLoader(
                test_dataset, batch_size=max_batch_size, shuffle=False)
        else:
            # create a DataLoader to load the data in batches
            train_dataset = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = torch.utils.data.DataLoader(
                test_dataset, batch_size=max_batch_size, shuffle=False)

        return train_dataset, test_dataset

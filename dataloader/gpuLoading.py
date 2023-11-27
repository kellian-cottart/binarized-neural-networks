import torch
import numpy as np
import idx2numpy


class GPUTensorDataset(torch.utils.data.Dataset):
    """ TensorDataset which has a data and a targets tensor and allows batching"""

    def __init__(self, data, targets, device="cuda:0", name=""):
        self.data = data.to(device)
        self.targets = targets.type(torch.LongTensor).to(device)
        self.name = name

    def __getitem__(self, index):
        """ Return a batch of data and targets """
        return self.data[index], self.targets[index]

    def __len__(self):
        """ Return the length of the dataset """
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.name)


class GPULoading:
    """ Load local datasets on GPU

    Args:
        batch_size (int): Batch size
        mean (float, optional): Mean of the dataset. Defaults to 0.1307.
        std (float, optional): Standard deviation of the dataset. Defaults to 0.3081.
        padding (int, optional): Padding to add to the images. Defaults to 0.
    """

    def __init__(self, batch_size, mean=0.1307, std=0.3081, padding=0, device="cuda:0", *args, **kwargs):
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.padding = padding
        self.device = device

    def __call__(self, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
        """ Load a local dataset

        Args:
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

        # add padding
        train_x = np.pad(train_x, ((0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        test_x = np.pad(test_x, ((0, 0), (self.padding, self.padding),
                                 (self.padding, self.padding)), 'constant')
        # if permute_idx is given, permute the dataset as for PermutedMNIST
        if "permute_idx" in kwargs:
            # permute_idx is the permutation to apply to the pixels of the images
            permute_idx = kwargs["permute_idx"]
            # Permute the pixels of the training examples
            train_x = train_x.reshape(
                train_x.shape[0], -1)[:, permute_idx].reshape(train_x.shape)
            # Permute the pixels of the test examples
            test_x = test_x.reshape(
                test_x.shape[0], -1)[:, permute_idx].reshape(test_x.shape)

        # apply normalisation
        train_x = (train_x - self.mean) / self.std
        test_x = (test_x - self.mean) / self.std

        # create a tensor which has self.data as data and self.targets as labels using the add_ids function
        train_dataset = GPUTensorDataset(torch.from_numpy(
            train_x), torch.from_numpy(train_y))
        test_dataset = GPUTensorDataset(torch.from_numpy(
            test_x), torch.from_numpy(test_y))

        # create a DataLoader to load the data in batches
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

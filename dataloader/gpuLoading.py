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


class GPULoading:
    """ Load local datasets on GPU

    Args:
        batch_size (int): Batch size
        mean (float, optional): Mean of the dataset. Defaults to 0.1307.
        std (float, optional): Standard deviation of the dataset. Defaults to 0.3081.
        padding (int, optional): Padding to add to the images. Defaults to 0.
    """

    def __init__(self, batch_size, padding=0, device="cuda:0", *args, **kwargs):
        self.batch_size = batch_size
        self.padding = padding
        self.device = device

    def __call__(self, path_train_x, path_train_y, path_test_x, path_test_y, turbo=True, *args, **kwargs):
        """ Load a local dataset

        Args:
            path_train_x (str): Path to the training data
            path_train_y (str): Path to the training labels
            path_test_x (str): Path to the testing data
            path_test_y (str): Path to the testing labels
            turbo (bool, optional): If False, loads a DataLoader, else runs tensors on GPU. Defaults to True.
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
        train_x = train_x.reshape(train_x.shape[0], -1) / 255
        test_x = test_x.reshape(test_x.shape[0], -1) / 255

        # Normalize the pixels in train_x and test_x using transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
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

        # create a tensor which has self.data as data and self.targets as targets
        train_dataset = GPUTensorDataset(train_x, torch.from_numpy(train_y))
        test_dataset = GPUTensorDataset(test_x, torch.from_numpy(test_y))

        # if we are using the turbo mode, we do not need to create a DataLoader
        if turbo:
            return train_dataset, test_dataset
        else:
            # create a DataLoader to load the data in batches
            train_dataset = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            max_test = len(test_dataset.data)
            test_dataset = torch.utils.data.DataLoader(
                test_dataset, batch_size=max_test, shuffle=False)

            return train_dataset, test_dataset

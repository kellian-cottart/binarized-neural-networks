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

    def __init__(self, path, batch_size, padding=0, num_workers=0, *args, **kwargs):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.padding = padding

    def __call__(self, dataset, *args, **kwargs):
        """ Load any dataset

        Args:
            dataset (torchvision.datasets): Dataset to load

        Returns:
            torch.utils.data.DataLoader: Training dataset
            torch.utils.data.DataLoader: Testing dataset
        """
        normalisation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x/255)
        ])

        train = dataset(root=self.path,
                        train=True,
                        download=True,
                        transform=normalisation,
                        target_transform=None)
        test = dataset(root=self.path,
                       train=False,
                       download=True,
                       transform=normalisation,
                       target_transform=None)

        train = DataLoader(train, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        max_test = len(test.data)
        test = DataLoader(test, batch_size=max_test,
                          shuffle=False, num_workers=self.num_workers)

        # pad the dataset to (28+padding*2)x(28+padding*2)
        if self.padding != 0:
            train.dataset.data = torch.nn.functional.pad(
                train.dataset.data, (self.padding, self.padding, self.padding, self.padding))
            test.dataset.data = torch.nn.functional.pad(
                test.dataset.data, (self.padding, self.padding, self.padding, self.padding))

        # if permute_idx is given, permute the dataset
        if "permute_idx" in kwargs and kwargs["permute_idx"] is not None:
            permute_idx = kwargs["permute_idx"]
            # Permute the pixels of the test examples
            train.dataset.data = train.dataset.data.reshape(
                train.dataset.data.shape[0], -1)[:, permute_idx].reshape(train.dataset.data.shape)
            test.dataset.data = test.dataset.data.reshape(
                test.dataset.data.shape[0], -1)[:, permute_idx].reshape(test.dataset.data.shape)

        return train, test

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
            root=self.path, download=True, transform=self.normalisation, target_transform=None, train=True)
        test = dataset(
            root=self.path, download=True, transform=self.normalisation, target_transform=None, train=False)

        train.data = train.data.to(torch.float32)
        test.data = test.data.to(torch.float32)

        # add padding
        padding = self.padding
        train.data = torch.nn.functional.pad(
            train.data, (padding, padding, padding, padding))
        test.data = torch.nn.functional.pad(
            test.data, (padding, padding, padding, padding))

        # if permute_idx is given, permute the dataset
        if "permute_idx" in kwargs:
            permute_idx = kwargs["permute_idx"]
            # Permute the pixels of the test examples
            train.data = train.data.reshape(
                train.data.shape[0], -1)[:, permute_idx].reshape(train.data.shape)
            test.data = test.data.reshape(
                test.data.shape[0], -1)[:, permute_idx].reshape(test.data.shape)
        train = DataLoader(
            train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test = DataLoader(
            test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train, test

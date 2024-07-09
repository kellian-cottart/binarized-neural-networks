
import torch


class GPUTensorDataset(torch.utils.data.Dataset):
    """ Dataset which has a data and a targets tensor, designed to be used on GPU

    Args:
        data (torch.tensor): Data tensor
        targets (torch.tensor): Targets tensor
        device (str, optional): Device to use. Defaults to "cuda:0".
        transform (torchvision.transforms.Compose, optional): Data augmentation transform. Defaults to None.
    """

    def __init__(self, data, targets, device="cuda:0"):
        self.data = data.to("cpu")
        self.targets = targets.to("cpu")
        self.device = device

    def __getitem__(self, index):
        """ Return a (data, target) pair """
        return self.data[index].to(self.device), self.targets[index].to(self.device)

    def __len__(self):
        """ Return the number of samples """
        return len(self.data)

    def shuffle(self):
        """ Shuffle the data and targets tensors """
        perm = torch.randperm(len(self.data), device="cpu")
        self.data = self.data[perm]
        self.targets = self.targets[perm]

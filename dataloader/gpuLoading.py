import torch
import numpy as np
import idx2numpy
from torchvision.transforms import v2
from torchvision import models, datasets
import os
import requests
import pickle
import numpy as np
import sys
from tqdm import tqdm
import hashlib
from .structures import *

PATH_MNIST_X_TRAIN = "datasets/MNIST/raw/train-images-idx3-ubyte"
PATH_MNIST_Y_TRAIN = "datasets/MNIST/raw/train-labels-idx1-ubyte"
PATH_MNIST_X_TEST = "datasets/MNIST/raw/t10k-images-idx3-ubyte"
PATH_MNIST_Y_TEST = "datasets/MNIST/raw/t10k-labels-idx1-ubyte"

PATH_FASHION_MNIST_X_TRAIN = "datasets/FashionMNIST/raw/train-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TRAIN = "datasets/FashionMNIST/raw/train-labels-idx1-ubyte"
PATH_FASHION_MNIST_X_TEST = "datasets/FashionMNIST/raw/t10k-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TEST = "datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte"

PATH_CIFAR10 = "datasets/cifar-10-batches-py"
PATH_CIFAR10_DATABATCH = [
    f"{PATH_CIFAR10}/data_batch_{i}" for i in range(1, 6)]
PATH_CIFAR10_TESTBATCH = f"{PATH_CIFAR10}/test_batch"

PATH_CIFAR100 = "datasets/cifar-100-python"
PATH_CIFAR100_DATABATCH = [f"{PATH_CIFAR100}/train"]
PATH_CIFAR100_TESTBATCH = f"{PATH_CIFAR100}/test"

REPOSITORY_CORE50_NPZ_128 = "http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz"
REPOSITORY_CORE50_PATHS = "https://vlomonaco.github.io/core50/data/paths.pkl"
REPOSITORY_CORE50_LABELS = "https://vlomonaco.github.io/core50/data/labels.pkl"
REPOSITORY_CORE50_LUP = "https://vlomonaco.github.io/core50/data/LUP.pkl"


class GPULoading:
    """ Load local datasets on GPU using the GPUTensorDataset

    Args:
        device (str, optional): Device to use. Defaults to "cuda:0".
    """

    def __init__(self, device="cuda:0", root="datasets", *args, **kwargs):
        self.device = device
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def task_selection(self, task, padding=0, *args, **kwargs):
        """ Select the task to load

        Args:
            task (str): Name of the task
        """
        self.padding = padding
        if "mnist" in task.lower():
            train, test = self.mnist(*args, **kwargs)
        elif "fashion" in task.lower():
            train, test = self.fashion_mnist(*args, **kwargs)
        elif "cifar100" in task.lower():
            train, test = self.cifar100(*args, **kwargs)
        elif "cifar10" in task.lower():
            train, test = self.cifar10(*args, **kwargs)
        elif "core50" in task.lower():
            scenario = task.split("-")[1]
            train, test = self.core50(
                scenario=scenario, run=0, *args, **kwargs)
        if isinstance(train, GPUTensorDataset):
            shape = train.data[0].shape
            target_size = len(train.targets.unique())
        else:
            shape = train[0].data[0].shape
            target_size = len(train[0].targets.unique())
        return train, test, shape, target_size

    def feature_extraction(self, folder, train_x, train_y, test_x, test_y, task="cifar100", iterations=10):
        """ Extract features using a resnet18 model

        Args:
            folder (str): Folder to save the features
            train_x (torch.tensor): Training data
            train_y (torch.tensor): Training labels
            test_x (torch.tensor): Testing data
            test_y (torch.tensor): Testing labels
            task (str, optional): Name of the task. Defaults to "cifar100".
            iterations (int, optional): Number of passes to make. Defaults to 10.
        """
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
            v2.feature_extraction(220, antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.0,), std=(1.0,))
        ])
        transform_test = v2.Compose([
            v2.feature_extraction(220, antialias=True),
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

    def to_dataset(self, train_x, train_y, test_x, test_y):
        """ Create a DataLoader to load the data in batches

        Args:
            train_x (torch.tensor): Training data
            train_y (torch.tensor): Training labels
            test_x (torch.tensor): Testing data
            test_y (torch.tensor): Testing labels
            batch_size (int): Batch size

        Returns:
            DataLoader, DataLoader: Training and testing DataLoader

        """
        train_dataset = GPUTensorDataset(
            train_x, torch.Tensor(train_y).type(
                torch.LongTensor), device=self.device)
        test_dataset = GPUTensorDataset(test_x, torch.Tensor(test_y).type(
            torch.LongTensor), device=self.device)
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

    def fashion_mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_FASHION_MNIST_X_TRAIN):
            datasets.FashionMNIST("datasets", download=True)
        return self.mnist_like(PATH_FASHION_MNIST_X_TRAIN, PATH_FASHION_MNIST_Y_TRAIN,
                               PATH_FASHION_MNIST_X_TEST, PATH_FASHION_MNIST_Y_TEST, *args, **kwargs)

    def mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_MNIST_X_TRAIN):
            datasets.MNIST("datasets", download=True)
        return self.mnist_like(PATH_MNIST_X_TRAIN, PATH_MNIST_Y_TRAIN,
                               PATH_MNIST_X_TEST, PATH_MNIST_Y_TEST, *args, **kwargs)

    def mnist_like(self, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
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
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def cifar10(self, iterations=10, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR10 """
        # Deal with the training data
        if not os.path.exists("datasets/CIFAR10/raw"):
            datasets.CIFAR10("datasets", download=True)
        path_databatch = PATH_CIFAR10_DATABATCH
        path_testbatch = PATH_CIFAR10_TESTBATCH
        if "feature_extraction" in kwargs and kwargs["feature_extraction"] == True:
            folder = "datasets/cifar10_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder) or not os.path.exists(f"{folder}/cifar10_{iterations}_features_train.pt"):
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
                self.feature_extraction(
                    folder, train_x, train_y, test_x, test_y, task="cifar10", iterations=iterations)
            train_x = torch.load(
                f"{folder}/cifar10_{iterations}_features_train.pt")
            train_y = torch.load(
                f"{folder}/cifar10_{iterations}_target_train.pt")
            test_x = torch.load(
                f"{folder}/cifar10_{iterations}_features_test.pt")
            test_y = torch.load(
                f"{folder}/cifar10_{iterations}_target_test.pt")
        else:
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
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def cifar100(self, iterations=10, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR100 """
        if not os.path.exists("datasets/CIFAR100/raw"):
            datasets.CIFAR10("datasets", download=True)
        if "feature_extraction" in kwargs and kwargs["feature_extraction"] == True:
            folder = "datasets/cifar100_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder) or not os.path.exists(f"{folder}/cifar100_{iterations}_features_train.pt"):
                path_databatch = PATH_CIFAR100_DATABATCH
                path_testbatch = PATH_CIFAR100_TESTBATCH
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
        else:
            path_databatch = PATH_CIFAR100_DATABATCH
            path_testbatch = PATH_CIFAR100_TESTBATCH
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
            # Normalize and pad the data
            train_x, test_x = self.normalization(train_x, test_x)
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def core50(self, scenario="ni", run=0, download=True, *args, **kwargs):
        return CORe50(scenario=scenario, run=run, download=download,
                      device=self.device).get_dataset()

### INSPIRED BY Vincenzo Lomonaco ###


class CORe50:
    """ Load the CORe50 dataset

    Args:
        root (str, optional): Root folder for the dataset. Defaults to "datasets".
        scenario (str, optional): Scenario to load. Defaults to "ni".
        run (int, optional): Run to load. Defaults to 0.
        start_batch (int, optional): Starting batch. Defaults to 0.
        download (bool, optional): Download the dataset. Defaults to True.
        device (str, optional): Device to use. Defaults to "cuda:0".
    """

    def __init__(self, root="datasets", scenario="ni", run=0, download=True, device="cuda:0"):
        self.root = os.path.join(root, "core50")
        self.scenario = scenario
        self.run = run
        self.device = device
        self.batch_scenario = {
            "ni": 8,
            'nc': 9,
            'nic': 79,
            'nicv2_79': 79,
            'nicv2_196': 196,
            'nicv2_391': 391
        }
        self.md5 = {
            "core50_imgs.npz": "3689d65d0a1c760b87821b114c8c4c6c",
            "labels.pkl": "281c95774306a2196f4505f22fd60ab1",
            "paths.pkl": "b568f86998849184df3ec3465290f1b0",
            "LUP.pkl": "33afc26faa460aca98739137fdfa606e"
        }
        if not os.path.exists(self.root) or not os.listdir(self.root):
            os.makedirs(self.root, exist_ok=True)
            self.download_dataset()

        bin_path = os.path.join(self.root, "core50_imgs.bin")
        if not os.path.exists(bin_path):
            data = np.load(os.path.join(self.root, "core50_imgs.npz"))['x']
            data.tofile(bin_path)

        self.data = np.fromfile(bin_path, dtype=np.uint8).reshape(
            164866, 128, 128, 3)
        self.labels = pickle.load(
            open(os.path.join(self.root, "labels.pkl"), "rb"))
        self.paths = pickle.load(
            open(os.path.join(self.root, "paths.pkl"), "rb"))
        self.lup = pickle.load(open(os.path.join(self.root, "LUP.pkl"), "rb"))

    def download_dataset(self):
        """ Download the dataset """
        files_to_download = [
            ("core50_imgs.npz", REPOSITORY_CORE50_NPZ_128),
            ("paths.pkl", REPOSITORY_CORE50_PATHS),
            ("labels.pkl", REPOSITORY_CORE50_LABELS),
            ("LUP.pkl", REPOSITORY_CORE50_LUP)
        ]
        for file_name, url in files_to_download:
            file_path = os.path.join(self.root, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                self.download_file(url, file_path)

    def checksum(self, file_path):
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(4096):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def download_file(self, url, file_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(file_path, 'wb') as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if not self.checksum(file_path) == self.md5[os.path.basename(file_path)]:
            print("Checksum failed. Deleting file.")
            os.remove(file_path)
            sys.exit(1)
        else:
            print("Checksum validated for " + file_path)

    def get_dataset(self):
        """ Returns the train and test sequential datasets"""

        test_indexes = self.lup[self.scenario][self.run][-1]
        test_x = torch.tensor(self.data[test_indexes]).float().to("cpu")
        test_x = test_x.permute(0, 3, 1, 2) / 255
        v2.Normalize((0,), (1,), inplace=True)(test_x, test_x)
        test_y = torch.tensor(
            self.labels[self.scenario][self.run][-1]).to("cpu")
        test_dataset = GPUTensorDataset(test_x, test_y, device=self.device)
        train_loader = []
        for i in range(self.batch_scenario[self.scenario]):
            train_indexes = self.lup[self.scenario][self.run][i]
            train_x = torch.tensor(self.data[train_indexes]).float().to("cpu")
            train_x = train_x.permute(0, 3, 1, 2) / 255
            train_y = torch.tensor(
                self.labels[self.scenario][self.run][i]).to("cpu")
            # normalize the data between 0 and 1
            v2.Normalize((0,), (1,), inplace=True)(
                train_x, train_x)
            train_loader.append(
                GPUTensorDataset(train_x, train_y, device=self.device))
        return train_loader, test_dataset

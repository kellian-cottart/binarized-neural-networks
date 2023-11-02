import torchvision.datasets as datasets

DATASETS_PATH = "./archive"
if __name__ == "__main__":
    mnist = datasets.MNIST(root=DATASETS_PATH, download=True)
    cifar = datasets.CIFAR10(root=DATASETS_PATH, download=True)

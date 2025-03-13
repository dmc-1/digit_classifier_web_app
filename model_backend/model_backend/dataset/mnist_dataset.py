import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MNISTDataset:
    def __init__(self, config):
        self.data_root = "data"

        self.config = config

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def build_train_dataloader(self):
        mnist_dataset = torchvision.datasets.MNIST(self.data_root,
                                                   transform=self.train_transforms,
                                                   download=True)
        data_loader = DataLoader(mnist_dataset, batch_size=self.config["train_batch_size"], shuffle=True)

        return data_loader

    def build_test_dataloader(self):
        mnist_dataset = torchvision.datasets.MNIST(self.data_root,
                                                   train=False,
                                                   transform=self.test_transforms,
                                                   download=True)
        data_loader = DataLoader(mnist_dataset, batch_size=self.config["test_batch_size"], shuffle=True)

        return data_loader

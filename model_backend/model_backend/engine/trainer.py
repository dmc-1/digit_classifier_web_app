import yaml

import torch
import torch.nn as nn
from model_backend.dataset.mnist_dataset import MNISTDataset
from model_backend.model.mini_resnet import MiniResNet


class Trainer:
    def __init__(self, config):
        self.device = 'cuda'

        mnist_dataset = MNISTDataset(config["dataset"])

        self.train_dataloader = mnist_dataset.build_train_dataloader()
        self.test_dataloader = mnist_dataset.build_test_dataloader()

        self.model = MiniResNet(config["model"])

        lr = config["optimiser"]["lr"]
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss()

        self.epochs = config["training"]["epochs"]

    def train(self):
        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            self.model.train()
            for images, labels in self.train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimiser.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()
                total += labels.size(0)
                correct += (torch.argmax(logits, 1) == labels).sum().item()

            train_loss = running_loss
            train_accuracy = correct / total

            running_loss = 0.0
            correct = 0.0
            total = 0.0
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.test_dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                    running_loss += loss.item()
                    total += labels.size(0)
                    correct += (torch.argmax(logits, 1) == labels).sum().item()

            val_loss = running_loss
            val_accuracy = correct / total

            print(f"Epoch {epoch}:\n"
                  f"train loss - {train_loss}, train acc - {train_accuracy}\n"
                  f"validation loss - {val_loss}, validation acc - {val_accuracy}")

    def save(self, save_path):
        self.model.to('cpu')
        torch.save(self.model, save_path)


if __name__ == "__main__":
    with open("../configs/config.yml", 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()
    trainer.save("../artifacts/model.pth")


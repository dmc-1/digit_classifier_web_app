import numpy as np
import torch
from torchvision import transforms

class Predictor:
    def __init__(self, model_path):
        self.resize_transform = transforms.Resize((28, 28))
        self.mnist_mean = 0.1307
        self.mnist_std = 0.3081

        self.model = torch.load(model_path)
        self.model.to('cpu')
        self.model.eval()

    def predict(self, img):
        img = self.process_image(img)
        logits = self.model(img).squeeze()
        probs = torch.softmax(logits, 0)

        label = torch.argmax(probs).item()
        confidence = probs[label].item()

        return label, confidence

    def process_image(self, img):
        img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
        img = self.resize_transform(img)
        img = (img - self.mnist_mean) / self.mnist_std

        return img
"""Small CNN that reads one cell (a letter, or the empty class)."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class CellCNN(nn.Module):
    # input: 1 x 32 x 32 grayscale
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def eval_transform(image_size):
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])


def train_transform(image_size, rotation_deg, brightness, contrast):
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=rotation_deg),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])


def save_checkpoint(path, model, classes, image_size):
    torch.save({"model_state": model.state_dict(), "classes": classes, "image_size": image_size}, path)


@dataclass
class CellPrediction:
    label: str
    probability: float


class CellClassifier:
    def __init__(self, checkpoint):
        ckpt = torch.load(checkpoint, weights_only=True, map_location="cpu")
        self.classes = list(ckpt["classes"])
        self.model = CellCNN(len(self.classes))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.transform = eval_transform(ckpt["image_size"])

    def predict(self, cells):
        """One CellPrediction per BGR cell image."""
        if not cells:
            return []
        # same conversion as at training time: BGR array -> RGB PIL image -> grayscale
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in cells]
        batch = torch.stack([self.transform(img) for img in images])
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1).numpy()
        best = probs.argmax(axis=1)
        return [CellPrediction(self.classes[i].upper(), float(p[i])) for p, i in zip(probs, best)]

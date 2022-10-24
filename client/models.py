import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.models import resnet18, squeezenet1_1
from torch.optim import Optimizer
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Compose, Normalize
from dataclasses import dataclass
from typing import Callable
from pathlib import Path


MODEL_DIR = Path(__file__).parent / "initial_models"
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class TrainingSettings:
    model: nn.Module
    steps: int
    optimizer: Optimizer
    lr: float
    loss: Callable
    train_ds: Dataset
    train_dl: DataLoader
    test_ds: Dataset
    test_dl: DataLoader
    target_accuracy: float


def get_models():
    return ["simple", "cnn", "resnet18", "squeeze_net"]


def get_training_settings(model_name="simple"):
    if model_name == "simple":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        model.load_state_dict(torch.load(MODEL_DIR / "initial-simple-model.pth"))
        steps = 100000
        lr = 0.04
        optimizer = torch.optim.SGD
        loss_fn = F.cross_entropy

        train_ds = MNIST(DATA_DIR, transform=ToTensor(), download=True)
        train_dl = DataLoader(train_ds, 256, shuffle=True)

        test_ds = MNIST(DATA_DIR, train=False, transform=ToTensor())
        test_dl = DataLoader(test_ds, 256)
        target_accuracy = 0.91
    elif model_name == "cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # bs x 16 x 16 x 16
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # bs x 16 x 8 x 8
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # bs x 16 x 4 x 4
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # bs x 16 x 2 x 2
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # bs x 16 x 1 x 1
            nn.Flatten(),  # bs x 16
            nn.Linear(16, 10),  # bs x 10
        )
        model.load_state_dict(torch.load(MODEL_DIR / "initial-cnn-model.pth"))
        steps = 5000000
        optimizer = torch.optim.SGD
        lr = 0.1
        loss_fn = nn.CrossEntropyLoss()
        transforms = Compose(
            [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        train_ds = CIFAR10(DATA_DIR, download=True, transform=transforms)
        train_dl = DataLoader(train_ds, 128, shuffle=True)

        test_ds = CIFAR10(DATA_DIR, train=False, transform=transforms)
        test_dl = DataLoader(test_ds, 128)
        target_accuracy = .91
    elif model_name == "resnet18":
        model = resnet18()
        model.load_state_dict(torch.load(MODEL_DIR / "initial-resnet18-model.pth"))
        steps = 1000
        optimizer = torch.optim.SGD
        lr = 0.1
        target_accuracy = .5
        loss_fn = nn.CrossEntropyLoss()
        transforms = Compose(
            [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        train_ds = CIFAR10(DATA_DIR, download=True, transform=transforms)
        train_dl = DataLoader(train_ds, 32, shuffle=True)

        test_ds = CIFAR10(DATA_DIR, train=False, transform=transforms)
        test_dl = DataLoader(test_ds, 32)
    elif model_name == "squeeze_net":
        model = squeezenet1_1()
        model.load_state_dict(torch.load(MODEL_DIR / "initial-squeeze1_1-model.pth"))
        steps = 100000
        optimizer = torch.optim.SGD
        lr = .01
        target_accuracy = .5
        loss_fn = nn.CrossEntropyLoss()
        transforms = Compose(
            [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
        )
        train_ds = CIFAR10(DATA_DIR, download=True, transform=transforms)
        train_dl = DataLoader(train_ds, 32, shuffle=True)

        test_ds = CIFAR10(DATA_DIR, train=False, transform=transforms)
        test_dl = DataLoader(test_ds, 32)
    return TrainingSettings(
        model=model,
        steps=steps,
        optimizer=optimizer,
        loss=loss_fn,
        lr=lr,
        train_ds=train_ds,
        train_dl=train_dl,
        test_ds=test_ds,
        test_dl=test_dl,
        target_accuracy=target_accuracy,
    )

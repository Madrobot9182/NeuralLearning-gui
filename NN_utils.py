"""
Utility file for both the app and learning models. Don't mind the pyramid of import statements ;)
"""

import torch
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

import NN_models


def loadDataset(set_name, batch_size=64, pin_mem=False, num_workers=0):
    match set_name:
        case "MNIST":
            dataset = datasets.MNIST
        case "FashionMNIST":
            dataset = datasets.FashionMNIST
        case "CIFAR10":
            dataset = datasets.CIFAR10
        case "CIFAR100":
            dataset = datasets.CIFAR100
        case _:
            raise Exception("Dataset Not Supported")  # Should never reach here

    """Loads the specified dataset and returns the train/test dataloader"""
    training_data = dataset(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = dataset(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem) # X.shape: N, C, H, W; Y.shape: N
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    return train_loader, test_loader


def display_learning(accuracies:list, losses:list, metas:list):
    """Graph the loss over each epoch"""
    if len(accuracies) == 0 or len(metas) == 0:
        raise Exception("You have not performed any training yet")

    plt.figure(figsize=(18, 12))
    plt.title("Model Performance", fontsize=48)
    plt.xlabel("Epoch", weight="normal", fontsize=32)
    plt.ylabel("Loss", weight="normal",fontsize=32)

    # Actually plot the data
    for accuracy, loss, meta in zip(accuracies, losses, metas):  # zip to correctly unpack each list
        model_name = meta["model_name"]
        optimizer_name = meta["optimizer_name"]
        epochs = meta["epochs"]

        epoch_line = np.linspace(0, epochs, epochs)
        plt.plot(epoch_line, accuracy, label=f"{model_name} with {optimizer_name} Accuracy")
        plt.plot(epoch_line, loss, label=f"{model_name} with {optimizer_name} Loss")

    # Save and show the graph
    plt.legend(loc="upper right", prop={"size": 12})
    plt.tight_layout()
    plt.savefig('Performance.png')
    plt.show()


def get_model(model):
    """Returns function to retrieve dataset"""
    return getattr(NN_models, model)  # NN_models.model_name


def get_loss(loss):
    """Returns function to retrieve dataset"""
    match loss:
        case "Mean Square Error":
            return nn.MSELoss
        case "Negative Log-Likelihood":
            return nn.NLLLoss
        case "Cross-Entropy":
            return nn.CrossEntropyLoss


def get_optimizer(optimizer):
    """Returns function to retrieve dataset"""
    match optimizer:
        case "Stochastic Gradient Descent":
            return optim.SGD
        case "Adagrad":
            return optim.Adagrad
        case "RMSprop":
            return optim.RMSprop


def get_device():
    """Simply gets the first available device, priorities the GPU"""
    device = ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")
    return device

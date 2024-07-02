'''
Code for training/testing NN using pytorch. Adopted from the pytorch docs, with my comments for learning
'''

import numpy as np
from torch import no_grad, float
from NN_utils import loadDataset


def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Send data to device (gpu) first
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        optimizer.zero_grad()  # reset gradients of model parameters (they add up by default, zero them out for backprop)
        loss.backward()  # Get, then deposit the gradients of loss w.r.t each parameter
        optimizer.step()  # adjust parameters

        # Funky stats
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with no_grad():
        for X, y in dataloader:
            # Send data to gpu first
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # Count total lost
            correct += (pred.argmax(1) == y).type(float).sum().item()  # Count how many times the model predicted correctly

    # Normalize by # batches
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def train_and_test(dataset, model, loss, optimizer, device, batch_size, learning_rate, epochs):
    train_loader, test_loader = loadDataset(dataset, batch_size, True, 0)

    # Load the model into the device
    model = model().to(device, non_blocking=True)

    # Initialize the loss function
    loss_fn = loss()

    # Encapsulates the optimization algorithm to update model weights. SGD = stochastic gradiant descent
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Train and loss logic
    accuracies, losses = np.empty(0, ), np.empty(0, )  # Empty np arrays
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # Train and test the model
        train_loop(train_loader, model, loss_fn, optimizer, device, batch_size)
        accuracy, test_loss = test_loop(train_loader, model, loss_fn, device)

        # Append the results to an array
        accuracies = np.append(accuracies, accuracy)
        losses = np.append(losses, test_loss)

    print("DONE")
    return accuracies, losses


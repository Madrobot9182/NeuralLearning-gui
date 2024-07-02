"""
TO BE REPLACED WITH A SMARTER METHOD FOR MAKING MODELS
"""

from torch.nn import (Module, Sequential, NLLLoss, CrossEntropyLoss,
                      Flatten, Linear, ReLU, LogSoftmax, Conv2d, MaxPool2d)

class MLP1(Module):
    """
    A densely-connected feedforward network with a single hidden layer of 512
    ReLU units.  Output layer is 10 fully-connected log softmax
    units (i.e., the output will be a vector of log-probabilities).
    """
    def __init__(self):
        super().__init__()
        # Input will be a tensor with dimension `(n_datapoints, n_channels, height, width)`
        # So `(64, 1, 30, 30)` in the case of MNIST dataset
        self.layers = Sequential(
            Flatten(),                                    # -> (64, 900)
            Linear(in_features=28*28, out_features=512),  # -> (64, 512)
            ReLU(),                                       # -> (64, 512)
            Linear(in_features=512, out_features=10),     # -> (64, 10)
            LogSoftmax(dim=1)                             # -> (64, 10)
            )

    def forward(self, X):
        logits = self.layers(X)
        return logits

class MLP2(Module):
    """
    A densely-connected feedforward network with a two hidden layers of 128
    and 64 ReLU units respectively.  Output layer is 10 fully-connected log softmax
    units (i.e., the output will be a vector of log-probabilities).
    """
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Flatten(),
            Linear(in_features=28*28, out_features=128),  # -> (64, 128)
            ReLU(),
            Linear(in_features=128, out_features=64),     # -> (128, 64)
            ReLU(),
            Linear(in_features=64, out_features=10),
            LogSoftmax(dim=1)
            )

    def forward(self, X):
        log_probs = self.layers(X)
        return log_probs


class CNN_MNIST(Module):
    """
    A convolutional neural network with the following layers:
    * A layer of 32 convolutional units with a kernel size of 5x5 and a stride of 1,1, with relu activation
    * A max-pooling layer with a pool size of 2x2 and a stride of 2,2.
    * A layer of 64 convolutional units with a kernel size of 5x5 and the default stride, with relu activation.
    * A max-pooling layer with a pool size of 2x2 and the default stride.
    * A `Flatten` layer (to reshape the image from a 2D matrix into a single long vector)
    * A layer of 512 fully-connected linear units with relu activation
    * A layer of 10 fully-connected linear units with log-softmax activation (the output layer)

    Output layer is 10 fully-connected log-softmax units (i.e., the output will
    be a vector of log-probabilities).
    """
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2)),
            Flatten(),
            Linear(in_features=1024, out_features=512),
            ReLU(),
            Linear(in_features=512, out_features=10),
            LogSoftmax(dim=1)
        )

    def forward(self, X):
        log_probs = self.layers(X)
        return log_probs


#  CIFAR MODELS

class CNN_CIFAR10(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
            Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2)),
            Flatten(),
            Linear(in_features=512, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=10),
            LogSoftmax(dim=1)
        )

    def forward(self, X):
        log_probs = self.layers(X)
        return log_probs


class CNN_CIFAR10(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
            Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2)),
            Flatten(),
            Linear(in_features=512, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=100),
            LogSoftmax(dim=1)
        )

    def forward(self, X):
        log_probs = self.layers(X)
        return log_probs
from PyQt6.QtWidgets import QWidget, QComboBox, QLineEdit, QPushButton, QTextEdit, QLabel, QVBoxLayout

import NN_utils
import NN_learning
import sys


class GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.accuracies = []  # List of accuracies (list of list)
        self.losses = []  # List of loss results (list of list)
        self.meta_info = []  # (List of dictionaries)

        self.initUI()

    def initUI(self):
        # Create dropdown options
        # Create labels and input boxes
        datasetLabel = QLabel("Dataset:")
        self.datasetCombo = QComboBox()
        self.datasetCombo.addItems(["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"])
        self.datasetCombo.currentTextChanged.connect(self.updateModels)  # For updating available models

        modelLabel = QLabel("Model:")
        self.modelCombo = QComboBox()
        self.modelCombo.addItems(["MLP1", "MLP2", "CNN_MNIST"])

        lossLabel = QLabel("Loss Function:")
        self.lossCombo = QComboBox()
        self.lossCombo.addItems(["Negative Log-Likelihood", "Cross-Entropy", "Mean Square Error"])

        optimizerLabel = QLabel("Optimizer:")
        self.optimizerCombo = QComboBox()
        self.optimizerCombo.addItems(["Stochastic Gradient Descent", "Adagrad", "RMSprop"])

        deviceLabel = QLabel("Device:")
        self.deviceCombo = QComboBox()
        self.deviceCombo.addItems(["auto", "cuda", "cpu"])

        batchSizeLabel = QLabel("Batch Size:")
        self.batchSizeEdit = QLineEdit()
        self.batchSizeEdit = QLineEdit("64")  # Default batch size is 32

        learningRateLabel = QLabel("Learning Rate:")
        self.learningRateEdit = QLineEdit()
        self.learningRateEdit = QLineEdit("0.001")  # Default learning rate is 0.001

        epochsLabel = QLabel("Epochs:")
        self.epochsEdit = QLineEdit()
        self.epochsEdit = QLineEdit("5")  # Default epochs is 10

        # Create buttons
        self.beginButton = QPushButton("BEGIN")
        self.beginButton.clicked.connect(self.beginTraining)

        self.displayResultButton = QPushButton("Display Result")
        self.displayResultButton.clicked.connect(self.displayResult)

        # Create output box
        self.outputText = QTextEdit()
        self.outputText.setReadOnly(True)

        # Redirect console output to text widget
        sys.stdout = OutputRedirector(self.outputText, sys.stdout)
        sys.stderr = OutputRedirector(self.outputText, sys.stderr)

        # Create a QVBoxLayout
        layout = QVBoxLayout()

        # Add labels and widgets to the layout
        layout.addWidget(datasetLabel)
        layout.addWidget(self.datasetCombo)
        layout.addWidget(modelLabel)
        layout.addWidget(self.modelCombo)
        layout.addWidget(lossLabel)
        layout.addWidget(self.lossCombo)
        layout.addWidget(optimizerLabel)
        layout.addWidget(self.optimizerCombo)
        layout.addWidget(deviceLabel)
        layout.addWidget(self.deviceCombo)
        layout.addWidget(batchSizeLabel)
        layout.addWidget(self.batchSizeEdit)
        layout.addWidget(learningRateLabel)
        layout.addWidget(self.learningRateEdit)
        layout.addWidget(epochsLabel)
        layout.addWidget(self.epochsEdit)
        layout.addWidget(self.beginButton)
        layout.addWidget(self.displayResultButton)
        layout.addWidget(self.outputText)

        # Set the layout for the widget
        self.setLayout(layout)

    def updateModels(self, dataset):
        # Clear the model combo box
        self.modelCombo.clear()

        # Get the compatible models for the selected dataset
        models = []  # TODO: Automate the selection process (Maybe a folder with python files for each dataset?)
        match dataset:
            case "MNIST" | "FashionMNIST":
                models = ["MLP1", "MLP2", "CNN_MNIST", "LinearFashionMNIST"]
            case "CIFAR10":
                models = ["CNN_CIFAR10"]
            case "CIFAR100":
                models = ["CNN_CIFAR100"]

        # Add the compatible models to the model combo box
        self.modelCombo.addItems(models)

    def beginTraining(self):
        self.outputText.insertPlainText("Beginning Training...")
        # Get selected options
        dataset = self.datasetCombo.currentText()
        model = self.modelCombo.currentText()
        loss = self.lossCombo.currentText()
        optimizer = self.optimizerCombo.currentText()
        device = self.deviceCombo.currentText()

        try:
            batch_size = int(self.batchSizeEdit.text())
            learning_rate = float(self.learningRateEdit.text())
            epochs = int(self.epochsEdit.text())
        except ValueError as e:
            print("Invalid numerical input, numbers only please~", file=sys.stderr)

        # Get the metadata while it's still in string format
        meta = build_meta_info(model, optimizer, epochs)

        # Convert before passing into the function
        model = NN_utils.get_model(model)
        loss = NN_utils.get_loss(loss)
        optimizer = NN_utils.get_optimizer(optimizer)
        device = app_get_device(device)

        # Call test_and_train function
        try:
            accuracy, loss = NN_learning.train_and_test(dataset, model, loss, optimizer, device, batch_size, learning_rate, epochs)
            self.accuracies.append(accuracy)
            self.losses.append(loss)
            self.meta_info.append(meta)
        except Exception as e:
            print(repr(e), file=sys.stderr)

    def displayResult(self):
        try:
            NN_utils.display_learning(self.accuracies, self.losses, self.meta_info)
        except Exception as e:
            print(repr(e), file=sys.stderr)


class OutputRedirector(object):
    # For printing the console into the textbox
    def __init__(self, text_widget, stream):
        self.text_widget = text_widget
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        text = text.rstrip('\n')  # Remove trailing newlines
        self.text_widget.append(text)

    def flush(self):
        pass


def app_get_device(device):
    """Gets selected device"""
    if device == "auto":
        return NN_utils.get_device()
    return device


def build_meta_info(model_name, optimizer_name, epochs):
    """Creates a Json (dict) to be printed
    FORMAT: model name | optimizer name | epochs trained with"""
    return {"model_name": model_name, "optimizer_name": optimizer_name, "epochs": epochs}
import numpy as np
import torch
import torchvision

from pydantic import BaseModel
from typing import Union


LOSS_FUNCTION = torch.nn.CrossEntropyLoss()


class CNNConfig(BaseModel):
    in_channels: int
    out_channels: int
    convolution_kernel_size: int
    stride: int
    padding: int 
    pooling_kernel_size: int
    dropout_rate: float


class TrainingConfig(BaseModel):
    n_epochs: int
    learning_rate: float


class CNN(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                kernel_size=args.convolution_kernel_size,
                stride=args.stride,
                padding=args.padding
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=args.pooling_kernel_size),
            torch.nn.Dropout(args.dropout_rate)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=args.out_channels,
                out_channels=args.out_channels * 2,
                kernel_size=args.convolution_kernel_size,
                stride=args.stride,
                padding=args.padding
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=args.pooling_kernel_size),
            torch.nn.Dropout(args.dropout_rate)
        )

        self.output = torch.nn.Linear(in_features=args.out_channels ** 4, out_features=args.out_channels * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class ModelInterface:
    def __init__(self, args) -> None:
        self.model = CNN()
        self.n_epochs = args.n_epochs
        self.loss_fn = LOSS_FUNCTION
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.learning_rate)

    def train_model(self, training_loader, validation_loader, trained_model_path: Union[str, None] = None, save_model: bool = False, verbose: bool = True):
        accuracies = []
        losses = []

        for epoch in range(self.n_epochs):
            self.model.train()
            for X_batch, y_batch in training_loader:
                y_predicted = self.model(X_batch)
                loss = self.loss_fn(y_predicted, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # validation
            self.model.eval()
            accuracy = 0
            count = 0
            for X_batch, y_batch in validation_loader:
                y_predicted = self.model(X_batch)
                accuracy += (torch.argmax(y_predicted, 1) == y_batch).float().sum()
                count += len(y_batch)
            accuracy = accuracy / count

            accuracies.append(accuracy)
            losses.append(loss)

            if verbose:
                print("Epoch %3d | Model accuracy: %.2f%% | Loss: %.2f%%" % (epoch, accuracy * 100, loss))

        if save_model:
            torch.save(self.model.state_dict(), trained_model_path)

        return np.array(accuracies)
    
    def test_model(self, test_loader, verbose: bool = True):
        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                test_output = self.model(images)
                y_predicted = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (y_predicted == labels).sum().item() / float(labels.size(0))

        if verbose:
            print("Test accuracy (%5): %.2%%" % (len(test_loader), accuracy * 100))

        return accuracy
    
    def load_model(self, model_class, saved_model_path):
        model = model_class
        model.load_state_dict(torch.load(saved_model_path))
        return model.eval()
    

transform = torch.transforms.Compose(
    [torch.transforms.ToTensor(),
     torch.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


arch_config = {
    "in_channels": 1,
    "out_channels": 16, 
    "convolution_kernel_size": 5,
    "stride": 1,
    "padding": 2,
    "pooling_kernel_size": 2,
    "dropout_rate": 0.25
}

training_config = {
    "n_epochs": 10,
    "learning_rate": 0.001
}

model_config = CNNConfig(**arch_config)
interface_config = TrainingConfig(**training_config)

cnn = CNN(args=model_config)
interface = ModelInterface(args=interface_config)

interface.train_model(training_loader=trainloader, 
                      validation_loader=valida)




    
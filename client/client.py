from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import os
import flwr as fl
import random
from torch.utils.data import Subset
from sklearn.metrics import f1_score
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    # Create subsets
    train_indices = random.sample(range(len(trainset)), 5000)
    test_indices = random.sample(range(len(testset)), 1000)
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)

    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True) #usar trainset em train_subset para dados completos
    testloader = DataLoader(test_subset, batch_size=32) #usar testset em test_subset para dados completos

    num_examples = {"trainset": len(train_subset), "testset": len(test_subset)}
    return trainloader, testloader, num_examples

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return loss, accuracy, f1

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and data
net = Net().to(DEVICE)
trainloader, testloader, num_examples = load_data()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        start_time = time.time()
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        end_time = time.time()
        computation_time = end_time - start_time
        return self.get_parameters(config={}), num_examples["trainset"], {"computation_time": computation_time}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, f1 = test(net, testloader)
        return float(loss), num_examples["testset"], {"loss" : float(loss), "accuracy": float(accuracy), "f1": float(f1)}
    

# Inicia el cliente
if __name__ == "__main__":
    fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient().to_client())
    #fl.client.start_client(server_address=os.environ['SERVER_IP'], client=CifarClient().to_client())
    #fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())

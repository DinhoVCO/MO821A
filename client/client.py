from collections import OrderedDict
import torch
import os
import flwr as fl
from sklearn.metrics import f1_score

from models import get_model
from load_partition import LoadDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 2))
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
MODEL_NAME = os.environ.get("MODEL", "mobilenet_v2")
BATCH_SIZE = 32
DATASET = os.environ.get("DATASET", "CIFAR10")
PARTITIONER = os.environ.get("PARTITIONER", "DIRICHELT")


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    all_labels = []
    all_preds = []
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    loss /= len(testloader.dataset)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return loss, accuracy, f1

net = get_model(MODEL_NAME).to(DEVICE)
trainloader, testloader, num_examples = LoadDataset(CLIENT_ID).select_dataset(DATASET, NUM_CLIENTS, BATCH_SIZE, PARTITIONER)

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, f1 = test(net, testloader)
        return float(loss), num_examples["testset"], {"loss" : float(loss), "accuracy": float(accuracy), "f1": float(f1)}
    

# Inicia el cliente
if __name__ == "__main__":
    #fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient().to_client())
    fl.client.start_client(server_address=os.environ['SERVER_IP'], client=CifarClient().to_client())
    #fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())

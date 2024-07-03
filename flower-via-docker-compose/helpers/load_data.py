import numpy as np
#import tensorflow as tf
from flwr_datasets import FederatedDataset
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module


def load_data(data_sampling_percentage=0.5, client_id=1, total_clients=2, partitioner_type="PARTITIONER"):
    """Load federated dataset partition based on client ID.

    Args:
        data_sampling_percentage (float): Percentage of the dataset to use for training.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.

    Returns:
        Tuple of arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    # Download and partition dataset
    #fds = FederatedDataset(dataset="cifar10", partitioners={"train": total_clients})

    ## Non-IID
    if partitioner_type == "DIRICHLET":
        partitioner = DirichletPartitioner(num_partitions=total_clients, partition_by="label",
                                           alpha=0.5, min_partition_size=10, self_balancing=True)
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    elif partitioner_type == "PARTITIONER":
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": total_clients})
    else:
        raise ValueError(f"Partitioner {partitioner_type} is not supported.")
    

    partition = fds.load_partition(client_id - 1, "train")
    partition.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    # Apply data sampling
    num_samples = int(data_sampling_percentage * len(x_train))
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    x_train, y_train = x_train[indices], y_train[indices]

    return (x_train, y_train), (x_test, y_test)

from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


class LoadDataset:

    def __init__(self, cid):
        self.cid = cid

    def load_partition(self,dataset_name, n_clients, batch_size,partitioner_type):
        if(partitioner_type=='DIRICHELT'):
            partitioner = DirichletPartitioner(num_partitions=n_clients, partition_by="label",
                                    alpha=0.5, min_partition_size=10,
                                    self_balancing=True)
            fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})
        elif(partitioner_type=='PARTITIONER'):
            fds = FederatedDataset(dataset=dataset_name, partitioners={"train": n_clients})
        else:
            raise ValueError(f"Partitioner {partitioner_type} is not supported.")
        
        def apply_transforms(batch):
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            batch["img"] = [transform(img) for img in batch["img"]]
            return batch

        partition = fds.load_partition(self.cid, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8, seed=42)
        trainloader = DataLoader(partition["train"], batch_size=batch_size)
        testloader = DataLoader(partition["test"], batch_size=batch_size)
        num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}
        
        return trainloader, testloader, num_examples

    def select_dataset(self, dataset_name, n_clients, batch_size,partitioner_type='DIRICHELT'):
        if dataset_name == 'MNIST':
            return self.load_partition("nmist",n_clients, batch_size,partitioner_type)
        elif dataset_name == 'CIFAR10':
            return self.load_partition("cifar10",n_clients, batch_size,partitioner_type)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
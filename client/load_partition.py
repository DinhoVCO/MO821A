from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


class LoadDataset:

    def __init__(self, cid):
        self.cid = cid

    def load_partition(self,dataset_name, n_clients, batch_size):

        partitioner = DirichletPartitioner(num_partitions=n_clients, partition_by="label",
                                   alpha=0.5, min_partition_size=10,
                                   self_balancing=True)
        fds = FederatedDataset(dataset=dataset_name, partitioners={"train": partitioner})

        #fds = FederatedDataset(dataset=dataset_name, partitioners={"train": n_clients})
        
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

    def select_dataset(self, dataset_name, n_clients, batch_size):
        if dataset_name == 'MNIST':
            return self.load_partition("nmist",n_clients, batch_size)
        elif dataset_name == 'CIFAR10':
            return self.load_partition("cifar10",n_clients, batch_size)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")


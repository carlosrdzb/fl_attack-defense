import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataloader, Subset
import numpy as np

#Define the transformation for preprocessing the images
transform  =   transforms.Compose([
    transforms.ToTensor(),                                      #Convert images to PyTorch tensors
    transforms.Normalize((0.1307,),(0.3081,))                   #Normalize MNIST dataset
])

#Function to load the dataset and split it among clients
def load_mnist(num_clients=10):
    """Loads and partitions MNIST dataset into multiple clients"""

    dataset =   MNIST(root='./data',    train=True, download=True,  transform=transform)

    #Partition dataset for clients
    indices =   np.arange(len(dataset))
    np.random.shuffle(indices)
    split_indices   =   np.array_split(indices, num_clients)

    #Create a subset dataset for each client
    client_datasets =   [Subset(dataset,    idx)    for idx in  split_indices]

    return client_datasets
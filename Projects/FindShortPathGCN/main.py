import torch
from math import ceil
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from dataset import FindShortPathDataset
from visualization import visualize

# TODO: use train loader and mini-batches
# TODO: 20% for testing
# TODO: no fixed amount of steps
# TODO: 10'000 dataset


# TODO 2: Load the data using dataLoader or smth like that

# Hyper parameters and Data setup
dataset = FindShortPathDataset(root="./data")

n_epoch = 2
total_samples = len(dataset)
batch_size = 20
n_iterations = ceil(total_samples/batch_size)
print(f"Samples = {total_samples}, Iterations = {n_iterations}")

dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
dataIter = iter(dataLoader)

# data = next(dataiter)
# print(data[0].x, "\n", data[0].y)

# Get info and visualization
visualize(dataset, False)

# Training loop


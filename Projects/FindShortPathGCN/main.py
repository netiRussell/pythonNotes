import os
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.manifold import TSNE
from dataset import FindShortPathDataset
from visualization import visualize

# TODO: use train loader and mini-batches
# TODO: 20% for testing
# TODO: no fixed amount of steps
# TODO: 10'000 dataset


# TODO 1: RESOLVE - /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch_geometric/data/dataset.py:169: UserWarning: Found floating-point labels while calling `dataset.num_classes`. Returning the number of unique elements. Please make sure that this is expected before proceeding.
# TODO 2: Load the data using dataLoader or smth like that

# Custom Data setup
dataset = FindShortPathDataset(root="./data")

# Get info and visualization
visualize(dataset, True)
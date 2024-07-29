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

def split_data(dataset, val_ratio):
  train_size = int(total_samples * (1.0 - val_ratio))
  validation_size = total_samples - train_size

  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return iter(trainLoader), iter(validationLoader)


"""
  ## Short path prediction algoritm by Ruslan Abdulin ##
  Code to be executed starts here.
"""

# -- Hyper parameters and Data setup --
dataset = FindShortPathDataset(root="./data")

n_epochs = 2
batch_size = 20
total_samples = len(dataset)
n_iterations = ceil(total_samples/batch_size)
# print(f"Samples = {total_samples}, Iterations = {n_iterations}")

trainIter, validIter = split_data( dataset=dataset, val_ratio=0.2)


# -- Get info and visualization --
visualize(dataset, False)


# -- Training loop --
for epoch in range(n_epochs):
  # One epoch
  for batch in trainIter:
    # One batch
    for i in range(batch_size):
      # One sample
      print(batch[i].x, batch[i].y)
    break
  break

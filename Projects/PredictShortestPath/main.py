import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler
from torch_geometric.loader import DataLoader
from math import ceil
from models.Transformer import Transformer
from visualization import visualize
from dataset import PredictShortestPathDataset

import sys # TODO: delete after debugging is done

def split_data(dataset, val_ratio, total_samples):
  train_size = int(total_samples * (1.0 - val_ratio))
  validation_size = total_samples - train_size

  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return trainLoader, validationLoader


"""
multiple layers of GCN, then out = GCN is passed to a transformer to generate a sequence
"""
# TODO: Make the transformer work with the graph data

# -- Data -- 
batch_size = 25
dataset = PredictShortestPathDataset(root="./data")
total_samples = len(dataset)
n_iterations = ceil(total_samples/batch_size)

trainLoader, validLoader = split_data( dataset=dataset, val_ratio=0.2, total_samples=total_samples)

# -- Visualize a single data sample --
visualize(dataset, False)

# -- Hyperparameters --
n_epochs = 1
num_nodes = 16 # TODO: make it dynamic
# TODO: try num_nodes as the size
src_size = num_nodes # num of features for input
target_size = num_nodes # num of features for output
d_model = 64
num_heads = 8
num_layers = 6
d_ff = 256
max_seq_length = num_nodes
dropout = 0.1

transformer = Transformer(src_size, target_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# size must be >= than 2nd arg
# target_size doesn't have to be equaled to src_size
# different from src size of target is okay

# -- Training --
# TODO: consider using nn.MSELoss()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()
losses = []

for epoch in range(1):
    # One epoch
    for cur_batch_index, batch in enumerate(trainLoader):
      # One batch
      optimizer.zero_grad()

      for i in range(len(batch)):
        # One sample
        x = batch[i].x.permute(1,0)
        y = batch[i].y.permute(1,0)

        output = transformer(x, y[:, :-1], batch[i].edge_index)
        loss = criterion(output.contiguous().view(-1, target_size), y[:, 1:].contiguous().view(-1))
        loss.backward()
      
      optimizer.step()
      print(f"Epoch: {epoch+1}, Batch: {cur_batch_index}, Loss: {loss.item()}")


# # -- Evaluation --
# transformer.eval()

# # Generate random sample validation data
# val_src_data = torch.randint(1, src_size, (64, max_seq_length))  # (batch_size, seq_length)
# val_tgt_data = torch.randint(1, target_size, (64, max_seq_length))  # (batch_size, seq_length)

# with torch.no_grad():

#     val_output = transformer(val_src_data, val_tgt_data[:, :-1])
#     val_loss = criterion(val_output.contiguous().view(-1, target_size), val_tgt_data[:, 1:].contiguous().view(-1))
#     print(f"Validation Loss: {val_loss.item()}")
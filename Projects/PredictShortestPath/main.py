import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler
from torch_geometric.loader import DataLoader
from math import ceil
from models.Transformer import Transformer
from visualization import visualize
from dataset import PredictShortestPathDataset
import matplotlib.pyplot as plt

import sys # TODO: delete after done with debugging

def split_data(dataset, val_ratio, total_samples):
  train_size = int(total_samples * (1.0 - val_ratio))
  validation_size = total_samples - train_size

  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return trainLoader, validationLoader

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
src_size = num_nodes # num of features for input
target_size = num_nodes+1 # num of features for output
d_model = 64
num_heads = 8
num_layers = 6
d_ff = 256
max_seq_length = num_nodes*2+1
dropout = 0.1

transformer = Transformer(src_size, target_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# -- Training --
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()
losses = []

for epoch in range(2):
    # One epoch
    for cur_batch_index, batch in enumerate(trainLoader):
      # One batch
      optimizer.zero_grad()
      
      loss = None
      for i in range(len(batch)):
        # One sample
        x = batch[i].x.permute(1,0)
        y = batch[i].y.permute(1,0)

        # Another approach:
        # tgt = torch.zeros(17, dtype=torch.int32).unsqueeze(0)
        # tgt[0][-1] = 1

        output = transformer(x, y, batch[i].edge_index)

        sys.exit("_______________________")
      
      optimizer.step()
      print(f"Epoch: {epoch+1}, Batch: {cur_batch_index}, Loss: {loss.item()}")

      if(cur_batch_index % 20 == 0):
        losses.append(loss.item())


# -- Visualization of loss curve --
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# -- Evaluation --
transformer.eval()

with torch.no_grad():
  success_rate = []

  for _, batch in enumerate(validLoader):
    
    for i in range(len(batch)):
      x = batch[i].x.permute(1,0)
      y = batch[i].y.permute(1,0)

      val_output = transformer(x, y[:, :-1], batch[i].edge_index)
      y_hat = torch.empty(0)
      y = y[:, 1:].view(-1)

      # Convert output into a tensor of nodes ids'
      for elem in val_output[0]:
        y_hat = torch.cat((y_hat, torch.argmax(elem).unsqueeze(0)), 0)

      # Evaluate:
      points = 0
      length = len(y_hat)
      for i in range(length):
        if(y_hat[i].item() == y[i].item()):
          points += 1

      if( length != 0 ):
        success_rate.append(points / length)

      # print(f"Recieved: {y_hat}")
      # print(f"Expected: {y}")

  print(f"Success percentage: {sum(success_rate) / len(success_rate) }")

# TODO: Make transformer predict length of an answer
# TODO: Implement loss function for it
# TODO: Adapt the evaluation phase 
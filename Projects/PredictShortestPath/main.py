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

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # TODO: after done with debugging, shuffle = True
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return trainLoader, validationLoader

# -- Data -- 
batch_size = 25
dataset = PredictShortestPathDataset(root="./data")
total_samples = len(dataset)
n_iterations = ceil(total_samples/batch_size)

trainLoader, validLoader = split_data( dataset=dataset, val_ratio=0.2, total_samples=total_samples)

# -- Visualize a single data sample --
visualize(dataset, num_nodes=36, run=False)

# -- Hyperparameters --
n_epochs = 10
num_nodes = 36 # TODO: make it dynamic
src_size = num_nodes # num of features for input
target_size = num_nodes+1 # num of features for output
d_model = 64
num_heads = 8
num_layers = 6
d_ff = 256
max_seq_length = num_nodes+2 # max tgt length
dropout = 0.1

transformer = Transformer(src_size, target_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# -- Training --
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()
losses = []

for epoch in range(n_epochs):
    # One epoch
    for cur_batch_index, batch in enumerate(trainLoader):
      # One batch
      optimizer.zero_grad()
      
      loss = None
      for i in range(len(batch)):
        # One sample
        x = batch[i].x.permute(1,0)
        y = torch.cat(( batch[i].y.permute(1,0), torch.tensor([[len(batch[i].x)]]) ), 1) # labels + eos

        output = transformer(x, y, batch[i].edge_index, train_status=True)
        
        # length output = length y; because train_status=True
        loss = criterion(output.contiguous(), y.contiguous()[0])
        loss.backward()

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

  for id_batch, batch in enumerate(validLoader):
    
    for i in range(len(batch)):
      x = batch[i].x.permute(1,0)
      y = torch.cat(( batch[i].y.permute(1,0), torch.tensor([[len(batch[i].x)]]) ), 1) # labels + eos

      output = transformer(x, y, batch[i].edge_index, train_status=True)

      # Check if the length of the output is correct
      if(len(y[0]) != len(output)):
        success_rate.append(0)
        continue
      
      # Compare elements from output and labels
      points = 0

      for i, elem in enumerate(output):
        if(torch.argmax(elem) == y[0][i]):
          points += 1

      # len(y[0]) is never 0 because y = labels + eos
      success_rate.append(points / len(y[0]))
    
    print(f"Evaluation is in the process... Current batch = {id_batch}")

  print(f"Success percentage: {(sum(success_rate) / len(success_rate)) * 100 }%")
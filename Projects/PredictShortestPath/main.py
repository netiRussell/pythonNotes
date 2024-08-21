import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler
from torch_geometric.loader import DataLoader
from math import ceil
from models.Transformer import Transformer
from visualization import visualizeGraph, visualizeLoss
from dataset import PredictShortestPathDataset
import matplotlib.pyplot as plt
import os

import sys # TODO: delete after done with debugging

def split_data(dataset, val_ratio, total_samples):
  train_size = int(total_samples * (1.0 - val_ratio))
  validation_size = total_samples - train_size

  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # TODO: after done with debugging, shuffle = True
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return trainLoader, validationLoader


def save_checkpoint(state, path='./savedGrads/checkpoint.pth.tar'):
    # Overwrite prev saving
    if os.path.isfile(path):
        os.remove(path)
    
    torch.save(state, path)

# -- Data -- 
# ! TODO: Switch to GPU for the remote training
# ! TODO: improve the model
batch_size = 100
dataset = PredictShortestPathDataset(root="./data")
total_samples = len(dataset)
n_iterations = ceil(total_samples/batch_size)

trainLoader, validLoader = split_data( dataset=dataset, val_ratio=0.2, total_samples=total_samples)

# -- Visualize a single data sample --
visualizeGraph(dataset, num_nodes=100, run=False)

# -- Hyperparameters --
n_epochs = 3
num_nodes = 100 # TODO: make it dynamic(option: through dataset.py as a param of PredictShortestPathDataset)
src_size = num_nodes # num of features for input
target_size = num_nodes+1 # num of features for output
d_model = 256
num_heads = 8
num_layers = 6
d_ff = 1024
max_seq_length = num_nodes+2 # max tgt length
dropout = 0.1

# -- Training --
transformer = Transformer(src_size, target_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# -- Load model & optimizer --
# checkpoint = torch.load(PATH)
# transformer.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']

transformer.train()
losses = []

for epoch in range(n_epochs):
    # One epoch
    for cur_batch_index, batch in enumerate(trainLoader):
      # One batch
      optimizer.zero_grad()
      temp_losses = []
      
      loss = None
      for i in range(len(batch)):
        # One sample
        x = batch[i].x.permute(1,0)
        y = torch.cat(( batch[i].y.permute(1,0), torch.tensor([[len(batch[i].x)]]) ), 1) # labels + eos
        y_flag = batch[i].imperfect_y_flag.item()

        output = transformer(x, y, batch[i].edge_index, train_status=True)
        
        # length output = length y; because train_status=True
        loss = criterion(output.contiguous(), y.contiguous()[0])
        temp_losses.append(loss.item())

        # Imperfect sample case
        if(y_flag == 1):
          loss = 0.15*loss

        loss.backward()
      
      optimizer.step()
      print(f"Epoch: {epoch+1}, Batch: {cur_batch_index}, Loss: {loss.item()}")

      losses.append((sum(temp_losses) / len(temp_losses)))

# Save progress
save_checkpoint({
            'epoch': n_epochs,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            })

# -- Visualization of loss curve --
visualizeLoss(losses, run=True)

# -- Evaluation --
transformer.eval()

with torch.no_grad():
  success_rate = []
  complete_success_rate = []

  for id_batch, batch in enumerate(validLoader):
    
    for i in range(len(batch)):
      x = batch[i].x.permute(1,0)
      y = torch.cat(( batch[i].y.permute(1,0), torch.tensor([[len(batch[i].x)]]) ), 1) # labels + eos
      y_flag = batch[i].imperfect_y_flag.item()

      # Imperfect sample; to be disregarded
      if( y_flag == 1 ):
        continue

      output = transformer(x, y, batch[i].edge_index, train_status=True)

      # Check if the length of the output is correct
      if(len(y[0]) != len(output)):
        success_rate.append(0)
        complete_success_rate.append(0)
        continue
      
      # Compare elements from output and labels
      points = 0

      for i, elem in enumerate(output):
        if(torch.argmax(elem) == y[0][i]):
          points += 1

      # len(y[0]) is never 0 because y = labels + eos
      success_rate.append(points / len(y[0]))
      complete_success_rate.append(int(points / len(y[0])))
    
    print(f"Evaluation is in the process... Current batch = {id_batch}")

  print(f"Success percentage (length is correct but not all elements must be the same): {(sum(success_rate) / len(success_rate)) * 100 }%")
  print(f"Complete success percentage (length and all elements are correct): {(sum(success_rate) / len(success_rate)) * 100 }%")
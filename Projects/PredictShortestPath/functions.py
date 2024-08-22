from torch.utils.data import BatchSampler
from torch_geometric.loader import DataLoader
import torch
import os

def split_data(dataset, val_ratio, total_samples, batch_size):
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
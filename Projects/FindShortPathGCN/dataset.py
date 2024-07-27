import os.path as osp

import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
import ast

class FindShortPathDataset(Dataset):
  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
      super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
      return "data.csv"

  @property
  def processed_file_names(self):
      return "none.pt"

  def download(self):
      pass

  def process(self):
    # ID for corresponding dataset 
    idx = 0

    # Grid 3x3 edge list
    edge_index = torch.tensor([
                        [0, 1, 1, 2, 0, 3, 1, 4, 2, 5, 3, 4, 4, 5, 3, 6, 4, 7, 5, 8, 6, 7, 7, 8],
                        [1, 0, 2, 1, 3, 0, 4, 1, 5, 2, 4, 3, 5, 4, 6, 3, 7, 4, 8, 5, 7, 6, 8, 7] 
                        ], dtype=torch.long)

    # Read the csv file
    self.df = pd.read_csv(self.raw_paths[0])
    self.df = self.df.reset_index()

    # For each row, create data and increment idx
    for index, row in self.df.iterrows():
        # Parameters for a dataset
        X = torch.tensor(ast.literal_eval(row['X']), dtype=torch.float)
        #y = torch.tensor([row['step1'], row['step2']], dtype=torch.float)

        # ! This y shape doesn't work with cross-entropy
        # Cross-entropy would require [0, 1, 0, 0, 1, 0, 0, 0, 0] shape
        y = X.tolist()
        y[int(row['step1'])] = [1]
        y[int(row['step2'])] = [1]
        y = torch.tensor(y, dtype=torch.float)
        
        data = Data(x=X, edge_index=edge_index, y=y, num_nodes=9)

        torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
        idx += 1

  def len(self):
      return len(self.processed_file_names)

  def get(self, idx):
      data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
      return data
  
# dataset = set of generated data.
# When dataset[some index] is accesed, self.get function called
# This function retrieves corresponding file from the data folder
# dataset = FindShortPathDataset(root="./data/customData")

# print(dataset[0].x)
# print(dataset[0].y)
# print(dataset[0].edge_index.t())

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
    self.df = pd.read_csv(self.raw_paths[0])
    return ['data_{}.pt'.format(i) for i in range(len(self.df)) ]
    #return 'none.pt'

  def download(self):
      pass

  def process(self):
    # ID for corresponding dataset 
    idx = 0

    # Read the csv file
    self.df = pd.read_csv(self.raw_paths[0])
    self.df = self.df.reset_index()

    # For each row, create data and increment idx
    for index, row in self.df.iterrows():
        # Parameters for a dataset
        X = torch.tensor(ast.literal_eval(row['X']), dtype=torch.float, requires_grad=True)
        y = torch.tensor(ast.literal_eval(row['Y']), dtype=torch.float)
        edge_index = torch.tensor(ast.literal_eval(row['Edge index']), dtype=torch.long)
        
        # TODO: make sure num_nodes is dynamic in later data stages
        data = Data(x=X, edge_index=edge_index, y=y, num_nodes=11)

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
# dataset = FindShortPathDataset(root="./data")

# print(dataset[0].x)
# print(dataset[0].y)
# print(dataset[0].edge_index.t())

# df = pd.read_csv("./data/raw/data.csv")
# df = df.reset_index()
# row = next(df.iterrows())
# print(row)

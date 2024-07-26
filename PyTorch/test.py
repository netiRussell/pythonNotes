# An example of dynamic output layer in GCN. Not sure if it is effective in learning
# Solution: transfer the weights to a new layer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class DynamicGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicGCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)  # Start with 1 output, will change dynamically

    def forward(self, x, edge_index, output_size):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        self.fc = nn.Linear(self.hidden_dim, output_size)
        return self.fc(x)

# Example usage
input_dim = 10    # Node feature dimension
hidden_dim = 16   # Hidden layer dimension
output_size = 3   # Desired output size

# Create a dummy graph with 5 nodes and random features
x = torch.randn(5, input_dim)         # Node features
edge_index = torch.tensor([[0, 1, 2, 3, 4, 0],   # Edges (source)
                           [1, 2, 3, 4, 0, 3]])  # Edges (target), as an example

data = Data(x=x, edge_index=edge_index)

model = DynamicGCN(input_dim, hidden_dim)
output = model(data.x, data.edge_index, output_size)

print(output)
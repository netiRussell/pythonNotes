# An example of dynamic output layer in GCN. Not sure if it learns at all
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class DynamicGCN(nn.Module):
    def __init__(self, hidden_dim):
        super(DynamicGCN, self).__init__()
        self.hidden_dim = hidden_dim
        # Initializing with placeholder values; these will be dynamically set in forward
        self.conv1 = None
        self.conv2 = None
        self.fc = None

    def forward(self, x, edge_index, output_size):
        input_dim = x.size(1)  # Get the input feature dimension dynamically
        if self.conv1 is None or self.conv1.in_channels != input_dim:
            self.conv1 = pyg_nn.GCNConv(input_dim, self.hidden_dim)
        if self.conv2 is None:
            self.conv2 = pyg_nn.GCNConv(self.hidden_dim, self.hidden_dim)
        if self.fc is None or self.fc.out_features != output_size:
            self.fc = nn.Linear(self.hidden_dim, output_size)
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)

# Example usage
hidden_dim = 16  # Hidden layer dimension
output_size = 3  # Desired output size

# Create a dummy graph with 5 nodes and random features
# Example with 10 input features
x1 = torch.randn(5, 10)  # Node features
edge_index1 = torch.tensor([[0, 1, 2, 3, 4, 0],   # Edges (source)
                            [1, 2, 3, 4, 0, 3]])  # Edges (target)

# Example with 8 input features
x2 = torch.randn(5, 8)  # Node features
edge_index2 = torch.tensor([[0, 1, 2, 3, 4, 0],   # Edges (source)
                            [1, 2, 3, 4, 0, 3]])  # Edges (target)

model = DynamicGCN(hidden_dim)

# Pass first graph through the model
output1 = model(x1, edge_index1, output_size)
print("Output with 10 input features:", output1)

# Pass second graph with a different input feature size
output2 = model(x2, edge_index2, output_size)
print("Output with 8 input features:", output2)

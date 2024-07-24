import os
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import FindShortPathDataset

# TODO: do citation dataset first
# TODO: use train loader and batches
# TODO: 20% for testing
# TODO: no fixed amount of steps
# TODO: 10'000 dataset

# Output = 2 nodes that represent step1 and step2
# Custom Data setup
dataset = FindShortPathDataset(root="./data/customData")

# Dataset info
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]

# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
# print(f'Contains self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')

# Visualization
# G = to_networkx(data, to_undirected=True)
# plt.figure(figsize=(7,7))
# plt.xticks([])
# plt.yticks([])
# nx.draw_networkx(G,
#                  pos=nx.spring_layout(G, seed=42),
#                  with_labels=False)
# plt.show()

# Model
class GCN(torch.nn.Module):
  def __init__(self):
      super(GCN, self).__init__()
      #torch.manual_seed(12345)
      self.conv1 = GCNConv(dataset.num_features, 4)
      self.conv2 = GCNConv(4, 2)
      self.classifier = Linear(2, 1)

  def forward(self, x, edge_index):
      h = self.conv1(x, edge_index)
      h = h.relu()
      h = self.conv2(h, edge_index)
      h = h.tanh()  # Final GNN embedding space.
      out = torch.sigmoid(self.classifier(h))# Apply a final (linear) classifier.
      return out, h

model = GCN()
print(model)
print(f"Input: {data.x}", '\n', f"Output: {data.y}")

# Lists to store loss and accuracy values
losses = []
accuracies = []
epochs = 50
criterion = torch.nn.BCELoss()  #Initialize the CrossEntropyLoss function.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


# Arbitrary check of the Network model
edge_index = torch.tensor([
                        [0, 1, 1, 2, 0, 3, 1, 4, 2, 5, 3, 4, 4, 5, 3, 6, 4, 7, 5, 8, 6, 7, 7, 8],
                        [1, 0, 2, 1, 3, 0, 4, 1, 5, 2, 4, 3, 5, 4, 6, 3, 7, 4, 8, 5, 7, 6, 8, 7] 
                        ], dtype=torch.long)
X = [[1], [0], [0],
     [0], [0], [0],
     [0], [1], [0]]
X = torch.tensor(X, dtype=torch.float)

y = [[1], [0], [0], 
     [1], [0], [0], 
     [1], [1], [0]]
y = torch.tensor(y, dtype=torch.float)

data = Data(x=X, edge_index=edge_index, y=y, num_nodes=9)
loss, h = train(data)
# print("Last loss: ", losses[-1])
print("Initial loss: ", loss)

# Training loop
# TODO: fix the problem with range >= 10
for i in range(7):
    data = dataset[i]
    for epoch in range(epochs):
        loss, h = train(data)
        losses.append(loss.item())

plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Arbitrary check of the Network model
data = Data(x=X, edge_index=edge_index, y=y, num_nodes=9)
loss, h = train(data)
# print("Last loss: ", losses[-1])
print("After training loss: ", loss)




import os
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#  Load Zachary's karate club network
dataset = KarateClub()
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Dataset info
data = dataset[0]
#print(f'data = {data.to_dict()}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
print(f'Contains self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# Visualization
# TODO: make the graph also show values
G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(7,7))
plt.xticks([])
plt.yticks([])
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=42),
                 with_labels=False,
                 node_color=data.y,
                 cmap="Set2")
plt.show()

# Model
class GCN(torch.nn.Module):
  def __init__(self):
      super(GCN, self).__init__()
      torch.manual_seed(12345)
      self.conv1 = GCNConv(dataset.num_features, 4)
      self.conv2 = GCNConv(4, 2)
      self.classifier = Linear(2, dataset.num_classes)

  def forward(self, x, edge_index):
      h = self.conv1(x, edge_index)
      h = h.tanh()
      h = self.conv2(h, edge_index)
      h = h.tanh()  # Final GNN embedding space.
      out = self.classifier(h)# Apply a final (linear) classifier.

      return out, h

model = GCN()
print(model)

# Lists to store loss and accuracy values
losses = []
accuracies = []
epochs = 100
criterion = torch.nn.CrossEntropyLoss()  #Initialize the CrossEntropyLoss function.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


# Training loop
for epoch in range(epochs):
    loss, h = train(data)
    losses.append(loss.item())

plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()




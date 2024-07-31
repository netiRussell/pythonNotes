import torch
from math import ceil
from torch_geometric.loader import DataLoader
from torch.nn import Linear, LeakyReLU, ELU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset import FindShortPathDataset
from visualization import visualize
import matplotlib.pyplot as plt

# TODO: Try some simple dataset to make sure the structure works

def split_data(dataset, val_ratio):
  train_size = int(total_samples * (1.0 - val_ratio))
  validation_size = total_samples - train_size

  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return trainLoader, validationLoader


"""
  ## Short path prediction algoritm by Ruslan Abdulin ##
  Code to be executed starts here.
"""

# -- Hyper parameters and Data setup --
dataset = FindShortPathDataset(root="./data")

n_epochs = 8
batch_size = 25
total_samples = len(dataset)
n_iterations = ceil(total_samples/batch_size)
# print(f"Samples = {total_samples}, Iterations = {n_iterations}")

trainLoader, validLoader = split_data( dataset=dataset, val_ratio=0.2)
#validIter = iter(validLoader)

# -- Get info and visualization --
visualize(dataset, False)


# -- Model --
class Network(torch.nn.Module):
  # Try hidden layers > 1 at the last layer

  def __init__(self, n_hidden_channels):
    super(Network, self).__init__()
    torch.manual_seed(1234567)
    self.conv1 = GCNConv(dataset.num_features, n_hidden_channels)
    self.conv2 = GCNConv(n_hidden_channels, n_hidden_channels)
    self.conv3 = GCNConv(n_hidden_channels, n_hidden_channels)
    self.conv4 = GCNConv(n_hidden_channels, 11)
    self.conv5 = GCNConv(11, 11)
    self.classifier = Linear(11, 1)
    self.elu = ELU(alpha=1.0, inplace=False)

  def forward(self, x, edge_index):
    out = self.conv1(x, edge_index)
    out = self.elu(out)
    out = F.dropout(out, p=0.2, training=self.training)
    out = self.conv2(out, edge_index)
    out = self.elu(out)
    out = F.dropout(out, p=0.5, training=self.training)
    out = self.conv3(out, edge_index)
    out = self.elu(out)
    out = F.dropout(out, p=0.5, training=self.training)
    out = self.conv4(out, edge_index)
    out = self.elu(out)
    out = F.dropout(out, p=0.5, training=self.training)
    out = self.conv5(out, edge_index)
    out = self.elu(out)
    out = F.dropout(out, p=0.5, training=self.training)
    out = self.classifier(out)
    return out

model = Network(n_hidden_channels=22)
print(model)

# -- Training loop --
losses = []
loss = None
# TODO: consider using Hamming Loss as a loss function
criterion = torch.nn.L1Loss()  # Initialize the loss function.
# TODO: consider some other than Adam optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

model.train()

X = torch.tensor([[-1], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0]], dtype=torch.float, requires_grad=True)# delete after testing 
y = torch.tensor([[0], [1], [4], [5], [-1], [-1], [-1], [-1], [-1], [-1], [-1]], dtype=torch.float)# delete after testing 
edge_index = torch.tensor([[0, 1, 0, 7, 1, 4, 7, 4, 1, 2, 4, 5, 7, 8, 4, 5, 5, 10, 2, 3, 3, 6, 6, 2, 8, 10, 6, 10, 9, 3], [1, 0, 7, 0, 4, 1, 4, 7, 2, 1, 5, 4, 8, 7, 5, 4, 10, 5, 3, 2, 6, 3, 2, 6, 10, 8, 10, 6, 3, 9]], dtype=torch.long)# delete after testing 

for epoch in range(n_epochs):
  # One epoch
  for cur_batch_index, batch in enumerate(trainLoader):
    # One batch
    optimizer.zero_grad()

    out = model(batch.x, batch.edge_index)
    loss = criterion(out, batch.y)
    loss.backward()

    optimizer.step()  # Update parameters based on gradients.

    if cur_batch_index % 25 == 0:
      losses.append(loss.item())
      print(f"Batch {cur_batch_index}, last loss = {losses[-1]:.4f}")
    
    if cur_batch_index % 100 == 0: # delete after testing 
      out = model(X, edge_index) # delete after testing 
      print("Result:", "\n", out, "\n", "Expected result:", "\n", y) # delete after testing 

# -- Validation -- 
# losses = []
# with torch.no_grad():
#   n_correct = 0
#   n_samples = 0
#   n_class_correct = [0 for i in range(10)]
#   n_class_samples = [0 for i in range(10)]
#   for images, labels in test_loader:
#       images = images.to(device)
#       labels = labels.to(device)
#       outputs = model(images)
#       # max returns (value ,index)
#       _, predicted = torch.max(outputs, 1)
#       n_samples += labels.size(0)
#       n_correct += (predicted == labels).sum().item()
      
#       for i in range(batch_size):
#           label = labels[i]
#           pred = predicted[i]
#           if (label == pred):
#               n_class_correct[label] += 1
#           n_class_samples[label] += 1

#   acc = 100.0 * n_correct / n_samples
#   print(f'Accuracy of the network: {acc} %')

#   for i in range(10):
#       acc = 100.0 * n_class_correct[i] / n_class_samples[i]
#       print(f'Accuracy of {classes[i]}: {acc} %')


# -- Visualization of loss curve --
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
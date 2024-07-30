import torch
from math import ceil
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from dataset import FindShortPathDataset
from visualization import visualize
import matplotlib.pyplot as plt

# TODO: use train loader and mini-batches
# TODO: 20% for testing
# TODO: no fixed amount of steps
# TODO: 10'000 dataset

def split_data(dataset, val_ratio):
  train_size = int(total_samples * (1.0 - val_ratio))
  validation_size = total_samples - train_size

  train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

  trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  validationLoader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

  return trainLoader, validationLoader

# TODO: apply mini-batching approach for training
def trainSample(sample, criterion):
  out = model(sample.x, sample.edge_index)
  loss = criterion(out, sample.y)
  loss.backward()

  return loss


"""
  ## Short path prediction algoritm by Ruslan Abdulin ##
  Code to be executed starts here.
"""

# -- Hyper parameters and Data setup --
dataset = FindShortPathDataset(root="./data")

n_epochs = 10
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
  def __init__(self, n_hidden_channels):
    super(Network, self).__init__()
    self.conv1 = GCNConv(dataset.num_features, n_hidden_channels)
    self.conv2 = GCNConv(n_hidden_channels, n_hidden_channels)
    self.conv3 = GCNConv(n_hidden_channels, n_hidden_channels)
    self.conv4 = GCNConv(n_hidden_channels, dataset.num_classes)
    self.classifier = Linear(dataset.num_classes, 1)

  def forward(self, x, edge_index):
    # TODO: Different activation functions
    out = self.conv1(x, edge_index)
    out = torch.tanh(self.conv2(out, edge_index))
    out = torch.nn.LeakyReLU(self.conv3(out, edge_index))
    out = torch.nn.LeakyReLU(self.conv4(out, edge_index))
    out = torch.nn.ELU(self.classifier(out))
    return out

model = Network(n_hidden_channels=3)
print(model)

# -- Training loop --
losses = []
loss = None
# TODO: consider using Hamming Loss as a loss function
criterion = torch.nn.MSELoss()  # Initialize the loss function.
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

for epoch in range(n_epochs):
  # One epoch
  for cur_batch_index, batch in enumerate(trainLoader):
    # One batch
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index)
    loss = criterion(out, batch.y)
    loss.backward()

    optimizer.step()  # Update parameters based on gradients.

    if cur_batch_index % 5 == 0:
      losses.append(loss.item())

    print(f"Batch {cur_batch_index}, last loss = {losses[-1]:.4f}")

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
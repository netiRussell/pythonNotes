import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Configurations
# device = torch.device ('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu') # cuda is not supported on Mac

# Hyper parameters
input_layer_size = 784 #images are 28x28
hidden_layer_size = 100
num_classes = 10 # 0-9 digits
num_epochs = 1
batch_size = 100 # number of samples
learning_rate = 0.01

# MNIST data
train_dataset = torchvision.datasets.MNIST(root = './data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root = './data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# -- Data examples --
# examples = iter(train_loader)
# samples, labels = examples.__next__()
# for i in range(9):
#   plt.subplot(3, 3, i+1)
#   plt.imshow(samples[i][0], cmap='gray')
# plt.show()

# Neural Network
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.l1 = nn.Linear(input_size, hidden_size) # 1st arg = input layer size, 2nd arg = output layer size
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, output_size) # output_size = num of classes
  
  def forward(self, current_sample):
    out = self.l1(current_sample)
    out = self.relu(out)
    out = self.l2(out)
    # No need for softmax at the end since crossentropy pyTorch function will be applied which includes the softmax in it.
    # (softmax is usually applied at the end of each multi-class classification NN)
    return out

model = NeuralNet(input_size=input_layer_size, hidden_size=hidden_layer_size, output_size=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    # forward
    output = model(images)
    loss = criterion(output, labels)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(i+1) % 100 == 0:
      print(f'epoch = {epoch+1}/{num_epochs}, step = {i+1}/{n_total_steps}, loss={loss.item():.4f}' )

# Evaluation of the training
with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    output = model(images)
  
    _, prediction = torch.max(output, 1) # returns values, index
    n_samples += labels.shape[0]
    n_correct += (prediction == labels).sum().item() # sums up all correct cases and converts the resulted tensor into regular int
  
  acc = 100 * n_correct / n_samples
  print(f'Accuracy = {acc:.4f}')




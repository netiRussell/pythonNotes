# PyTorch pipeline:
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer functions
# 3) Training loop
#   - Forward pass: compute prediction and loss
#   - Backword pass: compute gradients through backpropagation
#   - update weights:

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
n_samples = 20
n_features = 1
X_numpy, Y_numpy = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)


# 1) Design model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Construct loss and optimizer functions
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
  # Forward pass
  y_predicted = model(X)
  loss = criterion(y_predicted, Y)

  # Backword pass: compute gradients through backpropagation
  loss.backward()

  # Update weights:
  optimizer.step()

  # Clear the gradients
  optimizer.zero_grad()

  # Print
  if (epoch) % 10 == 0:
    print(f'epoch: {epoch}, loss = {loss.item():.4f}, ')


# Plot results
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
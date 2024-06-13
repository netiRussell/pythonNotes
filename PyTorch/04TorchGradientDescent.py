# Desired logic: f(x) = 2*x where 2 = w

import torch
import torch.nn as nn

# Data
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
nSamples, nFeatures = X.shape
test_val = torch.tensor([5.0], dtype=torch.float32)

# Model
class LinearRegression(nn.Module): 
  def __init__(self, input_dim, output_dim): 
    super (LinearRegression, self).__init__()

    # define layers 
    self.lin = nn.Linear (input_dim, output_dim, bias=False)

    # set initial weight value to zero
    with torch.no_grad():
      self.lin.weight[0,0] = 0
    
  def forward (self, x): 
    return self.lin(x)
  
model = LinearRegression (input_dim=nFeatures, output_dim=nFeatures,)


print(f'\nPrediction before training: f(5) = {model(test_val).item():.3f}' )

# Training
learningRate = 0.05
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
nIters = 20

for epoch in range(nIters):
  y_predicted = model(X)

  l = loss(Y, y_predicted)

  l.backward() # gradient is now calculated and stored at w.grad
  
  # update weights
  optimizer.step()
  
  # reset gradients
  optimizer.zero_grad()

  if epoch % 10 == 0:
    [w] = model.parameters()
    print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(test_val).item():.3f}\n' )
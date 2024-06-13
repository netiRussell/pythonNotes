import torch

# Desired logic: f(x) = 2*x where 2 = w
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model precition
def forward(x):
  return w*x

# loss = mean^2 error
def loss(y, y_predicted):
  return ((y_predicted - y)**2).mean()

print(f'\nPrediction before training: f(5) = {forward(5):.3f}' )

# Training
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
learningRate = 0.01
nIters = 20

for epoch in range(nIters):
  y_predicted = forward(X)

  l = loss(Y, y_predicted)

  l.backward() # gradient is now calculated and stored at w.grad
  
  # update weights
  with torch.no_grad():
    w -= learningRate * w.grad
  
  # reset gradients
  w.grad.zero_()

  print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}\n' )
import torch


# Initial values
w = torch.tensor(2.0, requires_grad=True) # parameter which needs to match y
learningRate = 0.01
neededResult = torch.tensor(16.0) # correct value

loss = 1.0
while( loss > 0.0001):
  # Forward pass
  result = 2*w # there can be a neuron instead of a 2.
  loss = (result-neededResult)**2

  # Backward pass
  loss.backward() # generates w.grad
  print(w.grad, loss)

  # Updating the weights
  w = torch.tensor(w.item() - learningRate*w.grad, requires_grad=True)

print("----------\nOptimal values is = ", w.item())
  






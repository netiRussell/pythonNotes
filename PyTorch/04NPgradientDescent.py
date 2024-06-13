import numpy as np

# Desired logic: f(x) = 2*x where 2 = w
w = 0.0

# model precition
def forward(x):
  return w*x

# loss = mean^2 error
def loss(y, y_predicted):
  return ((y_predicted - y)**2).mean()

# gradient
def gradient(x, y, y_predicted):
  return np.dot(2*x, y_predicted - y) / len(x) # dot product and calculating the mean

print(f'\nPrediction before training: f(5) = {forward(5):.3f}' )

# Training
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
learningRate = 0.01
nIters = 20

for epoch in range(nIters):
  y_predicted = forward(X)

  l = loss(Y, y_predicted)

  dw = gradient(X, Y, y_predicted)

  # update weights
  w -= learningRate * dw

  print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}\n' )
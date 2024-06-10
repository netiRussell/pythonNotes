import torch
import numpy as np


# Tesnor declaration and initialization -----------------------------------------------------------

# Create an empty 2x2 tensor
x = torch.empty(2,2)

# Create a 3x2 tensor with random values
x = torch.rand(3,2)

# Create a 1x1 tensor filled with zeros
x = torch.zeros(1)

# Create a 2x2 tensor filled with ones
x = torch.ones(2,2)

# Enable potential optimization feature
x = torch.rand(1,1, requires_grad=True)

# Specify data type
x = torch.ones(2,2, dtype=torch.int16)

# Convert list to a tensor
x = torch.tensor([[1,1,1], [1,1,1]])

# Convert 1x1 tensor into a value
x = torch.rand(1,1).item()


# Tesnor operations -------------------------------------------------------------------------------

# Add matrices (element addition)
x = torch.tensor([[1,1,1], [1,1,1]]) + torch.tensor([[2,1,2], [1,1,1]])

# Substract matrices (element substraction)
x = torch.tensor([[1,1,1], [1,1,1]]) - torch.tensor([[2,1,2], [1,1,1]])

# Single step convolution
x = torch.tensor([[3,4,6], [2,3,1]]) * torch.tensor([[1,0,1], [0,0,0]])

# Divide matrices (element division)
x = torch.tensor([[3,4,6], [2,3,1]]) / torch.tensor([[2,2,2], [2,2,2]])

# Slice operation
x = torch.rand(10, 8) # 10 rows 8 columns matrix
x = x[:, 0:5] # slice = 10 rows, 5 first column
x = x[0:3, 0:3] # slice = 3x3 as if for convolution

# Reshape tensor ( total # of elements must be the same )
x = torch.rand(4,4)
x = x.view(16)


print(x)


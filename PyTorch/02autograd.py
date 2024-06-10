import torch

# requires_grad=True enables the creation of backward functions every time some operation is applied to the matrix with such property. This backward function is needed for backpropogation.


# Get a gradient -----------------------------------------------------------------------------------
x = torch.randn(3, requires_grad=True)
print("X: ", x)

z = x+3
z = z.sum()
print("z=x+3: ",z)

z.backward() # x.grad = dz/dx

print("Resulted gradient: ", x.grad)


# Case where each iteration gradient must be reset -------------------------------------------------
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
  model_output = (weights*4).sum()
  model_output.backward()

  weights.grad.zero_()


# ! Every time you want to apply next operation, reset the gradient

# TODO: Get done this and backpropogation(next) sections and then write down notes regarding "requires_grad=True" part

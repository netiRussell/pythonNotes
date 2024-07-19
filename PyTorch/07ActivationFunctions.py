# Activation function - function that applies non-linear transformation to values and decides whether a neuron should be activated or not.
# Non-linearity improves precision and flexebility of a network

# Most popular activation functions are:
# - Sigmoid: transforms value into value between 0 and 1
# - TanH: sigmoid but outputs values between -1 and 1. Good choice for hidden layers.
# - ReLU: takes in values and outputs max(0, input). Universal choice for hidden layers.
# - Leaky ReLU: ReLU that if input < 0 would output a*input where "a" is some small negative number.
# - Softmax: takes in raw numbers and outputs probabilities from 0 to 1 which sum up to 1. See 06Softmax&crossEntropy.py for details.


import torch
import torch.nn as nn

# option 1: its own layer:
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(NeuralNet, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size) # 1st layer
    self.tanH = nn.Tanh() # 2nd layer
    self.linear2 = nn.Linear(hidden_size, 1) # 3rd layer
    self.sigmoid = nn.Sigmoid() # 4th layer

  def forward_pass(self, input_layer):
    output_layer = self.linear1(input_layer)
    output_layer = self.tanH(output_layer)
    output_layer = self.linear2(output_layer)
    output_layer = self.sigmoid(output_layer)
    return output_layer


# option 2: apply in forward pass
class NeuralNet2(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(NeuralNet, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size) # 1st layer
    self.linear2 = nn.Linear(hidden_size, 1) # 2nd layer

  def forward_pass(self, input_layer):
    output_layer = torch.tanh(self.linear1(input_layer))
    output_layer = torch.sigmoid(self.linear2(output_layer))
    return output_layer
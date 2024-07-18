import torch
import torch.nn as nn

# Softmax function takes in raw numbers and outputs probabilities from 0 to 1 which sum up to 1.
# For example: [2, 1, 0.1] => softmax => [0.65, 0.25, 0.1]

raw_data = torch.tensor([2,1,0.1])
output = torch.softmax(raw_data, dim=0)
print(f'Softmax with pytorch = {output}')

# Corss-entropy - function that measures the perfomance of a classification model
# Classification model - model that tries to apply a label to given data
# In pyTorch, cross entropy automatically applies softmax function
# If there is only two labels => classification is binary => use nn.BCELoss() instead
loss = nn.CrossEntropyLoss()

correct_label = torch.tensor([2]) # perfect data: value at index 2 is == 1 after softmax applied

# n_samples * n_classes = 1x3
raw_data_good = torch.tensor([[10.0, 20.0, 25.0]]) # probability of each label, index corresponds to the label
raw_data_bad = torch.tensor([[15.0, 30.0, 1.0]]) # probability of each label, index corresponds to the label

loss_good = loss(raw_data_good, correct_label)
loss_bad = loss(raw_data_bad, correct_label)
print(f'Cross-Entropy - good: {loss_good.item():.4f}, bad: {loss_bad.item():.4f}')
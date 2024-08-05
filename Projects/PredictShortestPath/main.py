import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from models.Transformer import Transformer

# TODO: Adapt the dataset.py
# TODO: Visualize data
# TODO: Split data
# TODO: Encode graph into passable value
# TODO: Make the transformer work with the graph data

# Hyperparameters
src_size = 5000
target_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_size, target_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, target_size, (64, max_seq_length))  # (batch_size, seq_length)

transformer = Transformer(src_size, target_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Training
# TODO: consider using nn.MSELoss()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

# for epoch in range(100):
#     optimizer.zero_grad()
#     output = transformer(src_data, tgt_data[:, :-1])
#     loss = criterion(output.contiguous().view(-1, target_size), tgt_data[:, 1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


# # Evaluation
# transformer.eval()

# # Generate random sample validation data
# val_src_data = torch.randint(1, src_size, (64, max_seq_length))  # (batch_size, seq_length)
# val_tgt_data = torch.randint(1, target_size, (64, max_seq_length))  # (batch_size, seq_length)

# with torch.no_grad():

#     val_output = transformer(val_src_data, val_tgt_data[:, :-1])
#     val_loss = criterion(val_output.contiguous().view(-1, target_size), val_tgt_data[:, 1:].contiguous().view(-1))
#     print(f"Validation Loss: {val_loss.item()}")
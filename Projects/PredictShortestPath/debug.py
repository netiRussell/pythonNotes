# This file is meant to be used as a training field for testing different algorithms

import torch

criterion = torch.nn.CrossEntropyLoss()

y = torch.tensor([[9, 5, 1, 0]]) 
y_hat = torch.tensor([[[ 0.1288, -0.2109,  0.7482, -0.8684,  0.5946,  10,  0.9755,
           0.6072, -0.3520,  0.2608,  0.1299,  0.2526,  0.4174,  0.0126,
           0.3552,  0.2631],
         [-0.1642,  2,  0.7412, -0.1880,  0.7510, -0.0440, -0.7743,
           0.4018, -0.6151, -0.2433,  0.3148,  0.7264,  0.5216,  0.0375,
           0.2543, -0.0227],
         [ 0, -0.0746,  0.7233,  0.4823,  0.3824, -0.3585, -0.6339,
          -1.1366,  0.4032,  0.6746, -0.2628,  0.9369, -0.0144,  0.1368,
           0.2331, -0.3543]]])

loss = criterion(y_hat.contiguous().view(-1, 16), y[:, 1:].contiguous().view(-1))
print(loss)
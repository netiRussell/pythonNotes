import numpy as np
import matplotlib.pyplot as plt

# x_data = np.random.random(50)
# y_data = np.random.random(50)
x_data = np.array([1,2], dtype=np.int8)
y_data = np.array([1,2], dtype=np.int8)

# Plot the given points
plt.scatter(x_data, y_data)

# Run this command to show the result of matplotlib
plt.show()
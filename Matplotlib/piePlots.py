import numpy as np
import matplotlib.pyplot as plt

x_data = ['Java', 'C++', 'JS', 'C', 'Rust']
y_data = [4, 15, 20, 7, 19]

# Plot the given points
plt.pie(y_data, labels=x_data)

# Run this command to show the result of matplotlib
plt.show()
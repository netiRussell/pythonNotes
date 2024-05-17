import numpy as np
import matplotlib.pyplot as plt

ages = np.random.normal(20, 1.5, 1000)
print(ages)

# Plot the given points
plt.hist(ages)

# Run this command to show the result of matplotlib
plt.show()
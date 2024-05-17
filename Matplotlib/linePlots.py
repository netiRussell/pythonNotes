import numpy as np
import matplotlib.pyplot as plt

years = [2005 + x for x in range(10)]
hours = [50,60,80,30,35,79,60,47,9,59]

# Plot the given points (x,y)
plt.plot(years, hours)

# Run this command to show the result of matplotlib
plt.show()
import numpy as np

# Create an array
array = np.arange(6)

# Create empty array full of zeros
zerosExmpl = np.zeros((3,3,3))

# Copy to get independent array with the same data
copyExmpl = array.copy()

# Convert list to an array and keep data as 8-bits int
array2 = np.array([1, 2, 3], dtype=np.int8)

# 2D arrays
array_2d = np.array([[1,2,3],[11,22,33]], dtype=np.uint8)
  # Or
array_2d = np.array([np.arange(6), np.arange(6)])

# See shape of the array(# of columns, rows, etc)
print("- 1D array: ", array, "shape: ", array.shape)
print("- 2D array:\n", array_2d, "shape: ", array_2d.shape)

# To reshape, use "reshape((rows, columns))"
print("\n\n- Reshaped 2D array as 3D:\n", array_2d.reshape((3,2,2)) )

# Embedded filter function
filter_array = np.arange(6)
filter_array[ filter_array < 4 ] = 0
print("\n\n- Array where every element less than 4 is set to 0: ", filter_array)
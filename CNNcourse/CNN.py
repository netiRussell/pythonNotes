import numpy as np
import h5py
import matplotlib.pyplot as plt
from public_tests import *

# Matplotlib parameters
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 1. Adding padding to a matrix -------------------------------------------------------------------------------
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    
    #(≈ 1 line)
    # X_pad = None
    # YOUR CODE STARTS HERE
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))
    
    # YOUR CODE ENDS HERE
    
    return X_pad

arr = np.array([1,1])
print(arr.shape)
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 3)
print ("x.shape =\n", x.shape)
print ("x_pad.shape =\n", x_pad.shape)
print ("x[1,1] =\n", x[1, 1])
print ("x_pad[1,1] =\n", x_pad[1, 1])

fig, axarr = plt.subplots(2, 2)

# first img before and after
axarr[0][0].set_title('x')
axarr[0][0].imshow(x[0, :, :, 0])
axarr[0][1].set_title('x_pad')
axarr[0][1].imshow(x_pad[0, :, :, 0])

# second img before and after
axarr[1][0].set_title('x')
axarr[1][0].imshow(x[1, :, :, 0])
axarr[1][1].set_title('x_pad')
axarr[1][1].imshow(x_pad[1, :, :, 0])

#plt.show()
zero_pad_test(zero_pad)
print("end of 1 --------------------------------")


# 2. Single Step of Convolution -------------------------------------------------------------------------------
def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    #(≈ 3 lines of code)
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * W
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    Z = Z.item()

    # Add bias b to Z. Cast b to a float().
    Z += float(b)

    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("\033[0;0mZ =", Z)
conv_single_step_test(conv_single_step)

print("end of 2 --------------------------------")


# 3. Forward Pass -------------------------------------------------------------------------------
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    f, f, n_C_prev, n_C = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters.get("stride")
    pad = hparameters.get("pad")
    
    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev + 2*pad - f)/stride + 1)
    n_W = int((n_W_prev + 2*pad - f)/stride + 1)
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range( m ):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]     # Select ith training example's padded activation
        for h in range( n_H ):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h
            vert_end = h + f
            
            for w in range( n_W ):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                horiz_start = w
                horiz_end = w + f
                
                for c in range( n_C ):   # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, : ]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    weights = W[i, 0:f, 0:f, c]
                    weights = weights[:, :, np.newaxis]
                    weights = np.repeat(weights, a_slice_prev.shape[2], axis=2)
                    print("Shape: ", weights.shape, "Weights: \n", weights)

                    biases = b[0, 0, 0, c ]

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

np.random.seed(1)
A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
z_mean = np.mean(Z)
z_0_2_1 = Z[0, 2, 1]
cache_0_1_2_3 = cache_conv[0][1][2][3]
print("Z's mean =\n", z_mean)
print("Z[0,2,1] =\n", z_0_2_1)
print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)

conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3)
conv_forward_test_2(conv_forward)

print("end of 3 --------------------------------")


# 3. Forward Pass -------------------------------------------------------------------------------

print("end of 4 --------------------------------")
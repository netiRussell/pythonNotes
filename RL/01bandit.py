# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from Bandits.rlglue.rl_glue import RLGlue
from Bandits import main_agent
from Bandits import ten_arm_env
from Bandits import test_env

def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    
    # Make sure q_values list is not empty
    if( len(q_values) == 0 ):
       return -1

    # initial values
    top_value = q_values[0]
    ties = []
    
    for i in range(len(q_values)):
      # if a value in q_values is greater than the highest value update top and reset ties to zero
      if( q_values[i] > top_value ):
         ties = []
         top_value = q_values[i]
         ties.append(i)
      
      # if a value is equal to top value add the index to ties
      elif( q_values[i] == top_value ):
         ties.append(i)
    
    # return a random selection from ties.   
    return np.random.choice(ties)

# -----------
# Tested Cell
# -----------
# The contents of the cell will be tested by the autograder.
# If they do not pass here, they will not pass there.

test_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
assert argmax(test_array) == 8, "Check your argmax implementation returns the index of the largest value"

# set random seed so results are deterministic
np.random.seed(0)
test_array = [1, 0, 0, 1]

counts = [0, 0, 0, 0]
for _ in range(100):
    a = argmax(test_array)
    counts[a] += 1

# make sure argmax does not always choose first entry
assert counts[0] != 100, "Make sure your argmax implementation randomly choooses among the largest values."

# make sure argmax does not always choose last entry
assert counts[3] != 100, "Make sure your argmax implementation randomly choooses among the largest values."

# make sure the random number generator is called exactly once whenver `argmax` is called
expected = [44, 0, 0, 56] # <-- notice not perfectly uniform due to randomness
assert counts == expected
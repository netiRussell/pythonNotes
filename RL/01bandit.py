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
         ties = [i]
         top_value = q_values[i]
      
      # if a value is equal to top value add the index to ties
      elif( q_values[i] == top_value ):
         ties.append(i)
    
    # return a random selection from ties.   
    return np.random.choice(ties)

class GreedyAgent(main_agent.Agent):
   def agent_step(self, reward, observation=None):
      """
      Takes one step for the agent. It takes in a reward and observation and 
      returns the action the agent chooses at that time step.
      
      Arguments:
      reward -- float, the reward the agent recieved from the environment after taking the last action.
      observation -- float, the observed state the agent is in

      Returns:
      current_action -- int, the action chosen by the agent at the current time step.
      """
      ### Useful Class Variables ###
      # self.q_values : An array with what the agent believes each of the values of the arm are.
      # self.arm_count : An array with a count of the number of times each arm has been pulled.
      # self.last_action : The action that the agent took on the previous time step
      #######################
      
      # increment the counter in self.arm_count for the action from the previous time step
      self.arm_count[self.last_action] += 1

      # update the step size using self.arm_count
      stepSize = 1.0 / self.arm_count[self.last_action]

      # update self.q_values for the action from the previous time step
      self.q_values[self.last_action] = self.q_values[self.last_action] + stepSize*(reward - self.q_values[self.last_action])

      
      
      # current action
      current_action = argmax( self.q_values )
      self.last_action = current_action
      
      return current_action
   
# ---------------
# Discussion Cell
# ---------------

num_runs = 200                    # The number of times we run the experiment
num_steps = 1000                  # The number of pulls of each arm the agent takes
env = ten_arm_env.Environment     # We set what environment we want to use to test
agent = GreedyAgent               # We choose what agent we want to use
agent_info = {"num_actions": 10}  # We pass the agent the information it needs. Here how many arms there are.
env_info = {}                     # We pass the environment the information it needs. In this case nothing.

rewards = np.zeros((num_runs, num_steps))
average_best = 0
for run in tqdm(range(num_runs)):           # tqdm is what creates the progress bar below
    np.random.seed(run)
    
    rl_glue = RLGlue(env, agent)          # Creates a new RLGlue experiment with the env and agent we chose above
    rl_glue.rl_init(agent_info, env_info) # We pass RLGlue what it needs to initialize the agent and environment
    rl_glue.rl_start()                    # We start the experiment

    average_best += np.max(rl_glue.environment.arms)
    
    for i in range(num_steps):
        reward, _, action, _ = rl_glue.rl_step() # The environment and agent take a step and return
                                                 # the reward, and action taken.
        rewards[run, i] = reward

greedy_scores = np.mean(rewards, axis=0)
plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot([average_best / num_runs for _ in range(num_steps)], linestyle="--")
plt.plot(greedy_scores)
plt.legend(["Best Possible", "Greedy"])
plt.title("Average Reward of Greedy Agent")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()
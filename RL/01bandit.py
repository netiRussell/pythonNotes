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

      
      
      # Selecting an action
      current_action = argmax( self.q_values )
      self.last_action = current_action
      
      return current_action


class EpsilonGreedyAgent(main_agent.Agent):
   def agent_step(self, reward, observation):
      """
      Takes one step for the agent. It takes in a reward and observation and 
      returns the action the agent chooses at that time step.
      
      Arguments:
      reward -- float, the reward the agent recieved from the environment after taking the last action.
      observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                           until future lessons
      Returns:
      current_action -- int, the action chosen by the agent at the current time step.
      """
      
      ### Useful Class Variables ###
      # self.q_values : An array with what the agent believes each of the values of the arm are.
      # self.arm_count : An array with a count of the number of times each arm has been pulled.
      # self.last_action : The action that the agent took on the previous time step
      # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
      #######################
      
      # increment the counter in self.arm_count for the action from the previous time step
      self.arm_count[self.last_action] += 1

      # update the step size using self.arm_count
      stepSize = 1.0 / self.arm_count[self.last_action]

      # update self.q_values for the action from the previous time step
      self.q_values[self.last_action] = self.q_values[self.last_action] + stepSize*(reward - self.q_values[self.last_action])
      
      
      # Selecting an action using epsilon greedy
      # Randomly choose a number between 0 and 1 and see if it's less than self.epsilon
      if( np.random.random() < self.epsilon ):
         # If it is, set current_action to a random action (explore).
         current_action = np.random.randint(0, len(self.q_values))
      else:
         # otherwise choose current_action greedily as you did above (exploit).
         current_action = argmax( self.q_values )
      
      
      self.last_action = current_action
      
      return current_action

class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
   def agent_step(self, reward, observation):
      """
      Takes one step for the agent. It takes in a reward and observation and 
      returns the action the agent chooses at that time step.
      
      Arguments:
      reward -- float, the reward the agent recieved from the environment after taking the last action.
      observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                           until future lessons
      Returns:
      current_action -- int, the action chosen by the agent at the current time step.
      """
      
      ### Useful Class Variables ###
      # self.q_values : An array with what the agent believes each of the values of the arm are.
      # self.arm_count : An array with a count of the number of times each arm has been pulled.
      # self.last_action : An int of the action that the agent took on the previous time step.
      # self.step_size : A float which is the current step size for the agent.
      # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
      #######################
      
      # increment the counter in self.arm_count for the action from the previous time step
      self.arm_count[self.last_action] += 1

      # Step size is constant

      # update self.q_values for the action from the previous time step
      self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size*(reward - self.q_values[self.last_action])
      
      
      # Selecting an action using epsilon greedy
      # Randomly choose a number between 0 and 1 and see if it's less than self.epsilon
      if( np.random.random() < self.epsilon ):
         # If it is, set current_action to a random action (explore).
         current_action = np.random.randint(0, len(self.q_values))
      else:
         # otherwise choose current_action greedily as you did above (exploit).
         current_action = argmax( self.q_values )
      
      
      self.last_action = current_action
      
      return current_action
   

env = ten_arm_env.Environment
env_info = {}

# ---------------
# Discussion Cell
# ---------------
epsilon = 0.1
num_steps = 2000
num_runs = 500
step_size = 0.1

plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='k')
plt.plot([1.55 for _ in range(num_steps)], linestyle="--")

for agent in [EpsilonGreedyAgent, EpsilonGreedyAgentConstantStepsize]:
    rewards = np.zeros((num_runs, num_steps))
    for run in tqdm(range(num_runs)):
        agent_info = {"num_actions": 10, "epsilon": epsilon, "step_size": step_size}
        np.random.seed(run)
        
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()

        for i in range(num_steps):
            reward, state, action, is_terminal = rl_glue.rl_step()
            rewards[run, i] = reward
            if i == 1000:
                rl_glue.environment.arms = np.random.randn(10)
        
    plt.plot(np.mean(rewards, axis=0))
plt.legend(["Best Possible", "1/N(A)", "0.1"])
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.show()
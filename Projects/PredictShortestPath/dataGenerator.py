import random
import pandas as pd
import torch
import torch_geometric.utils as tg
from collections import deque
import sys
import math

# Algorithm for finding the optimal path
def bfs(adjacency_matrix, S, par, dist):
  # Preparing graph by convertin long tesnor into int list 
  graph = adjacency_matrix

  # Queue to store the nodes in the order they are visited
  q = deque()
  # Mark the distance of the source node as 0
  dist[S] = 0
  # Push the source node to the queue
  q.append(S)

  # Iterate until the queue is not empty
  while q:
      # Pop the node at the front of the queue
      node = q.popleft()

      # Explore all the neighbors of the current node
      for neighbor in graph[node]:
          # Check if the neighboring node is not visited
          if dist[neighbor] == float('inf'):
              # Mark the current node as the parent of the neighboring node
              par[neighbor] = node
              # Mark the distance of the neighboring node as the distance of the current node + 1
              dist[neighbor] = dist[node] + 1
              # Insert the neighboring node to the queue
              q.append(neighbor)
    
  return dist


def get_shortest_distance(adjacency_matrix, S, D, V):
  # par[] array stores the parent of nodes
  par = [-1] * V

  # dist[] array stores the distance of nodes from S
  dist = [float('inf')] * V

  # Function call to find the distance of all nodes and their parent nodes
  dist = bfs(adjacency_matrix, S, par, dist)

  if dist[D] == float('inf'):
      print("Source and Destination are not connected")
      return

  # List path stores the shortest path
  path = []
  current_node = D
  path.append(D)
  while par[current_node] != -1:
      path.append(par[current_node])
      current_node = par[current_node]

  # Printing path from source to destination
  return path


# Global parameters
# n = random.randint(0,100)
# n must be = x^2
num_nodes = 16
if(math.sqrt(num_nodes) % 1):
    sys.exit(f"Number of nodes = {num_nodes} can't form a grid layout")



def generate_dataset(n, num_nodes):
    dataset = []
    for _ in range(n):
        # Generating random source and destination nodes ---------------------
        source = random.randint(0, num_nodes-1)
        destination = random.randint(0, num_nodes-1)
        
        # Find optimal path --------------------------------------------------

        # Dynamically generating edge_index
        edge_index = [[], []]
        n_rows = int(math.sqrt(num_nodes)) # n_rows = num of columns = number of elems per row

        for row in range(n_rows):
            for elem in range(n_rows):
                current_elem = elem + row*n_rows
                # lower neighbor
                if( current_elem + n_rows < num_nodes ):
                    edge_index[0].append(current_elem)
                    edge_index[1].append(current_elem+n_rows)
                    edge_index[0].append(current_elem+n_rows)
                    edge_index[1].append(current_elem)
                # right neighbor
                if( elem + 1 < n_rows ):
                    edge_index[0].append(current_elem)
                    edge_index[1].append(current_elem+1)
                    edge_index[0].append(current_elem+1)
                    edge_index[1].append(current_elem)
                # left neighbor
                if( elem - 1 > 0 ):
                    edge_index[0].append(current_elem)
                    edge_index[1].append(current_elem-1)
                    edge_index[0].append(current_elem-1)
                    edge_index[1].append(current_elem)
                # upper neighbor
                if( current_elem - n_rows > 0 ):
                    edge_index[0].append(current_elem)
                    edge_index[1].append(current_elem-n_rows)
                    edge_index[0].append(current_elem-n_rows)
                    edge_index[1].append(current_elem)

        # List to store the graph as an adjacency list
        graph = [[] for _ in range(num_nodes)]

        for i, node in enumerate(edge_index[0]):
            graph[node].append(edge_index[1][i])

        path = get_shortest_distance(graph, source, destination, num_nodes)
        
        # Generating X, and Y ----------------------------------------------------
        # X = nx1 size list of values of each node
        X = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        X[source] = [-1]
        X[destination] = [2]

        # Y = nodes to go to to reach destination. Minimum size: 1
        # path is reversed to follow s->d route
        # TODO: make Y be dynamic, so that Y = path[::-1]
        Y = [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]
        path = path[::-1]
        for i in range(len(path)):
            Y[i][0]= path[i]
        
        dataset.append([edge_index, X, Y])
    return dataset

# Generate a dataset
dataset = generate_dataset(10000, num_nodes)

# Create a DataFrame
df = pd.DataFrame(dataset, columns=["Edge index", "X", "Y"])

# Write the DataFrame to an CSV file
df.to_csv("./data/raw/data.csv", index=False)
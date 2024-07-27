import random
import pandas as pd
import torch
import torch_geometric.utils as tg
from collections import deque

# Data stages (Start and end points are always random):
    # 1) Static, same structure
    # 2) Semi-dynamic, structure is dynamic but it is derived from 10x10 grid
    # 3) Fully dynamic, structure is dynamic and is not limited in size

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
num_nodes = 11



def generate_dataset(n, num_nodes):
    dataset = []
    for _ in range(n):
        # Generating random source and destination nodes ---------------------
        source = random.randint(0, num_nodes-1)
        destination = random.randint(0, num_nodes-1)
        
        # Find optimal path --------------------------------------------------
        # TODO: edge_index must be dynamically initialized in the stages 2 and 3
        edge_index = torch.tensor([
                                [0, 1, 0, 7, 1, 4, 7, 4, 1, 2, 4, 5, 7, 8, 4, 5, 5, 10, 2, 3, 3, 6, 6, 2, 8, 10, 6, 10, 9, 3],
                                [1, 0, 7, 0, 4, 1, 4, 7, 2, 1, 5, 4, 8, 7, 5, 4, 10, 5, 3, 2, 6, 3, 2, 6, 10, 8, 10, 6, 3, 9] 
                                ], dtype=torch.long)

        # List to store the graph as an adjacency list
        graph = [[] for _ in range(num_nodes)]

        for i, node in enumerate(edge_index[0]):
            graph[node.item()].append(edge_index[1][i].item())

        path = get_shortest_distance(graph, source, destination, num_nodes)

        # TODO 3: Initialize Y
        
        # Generating X, and Y ----------------------------------------------------
        # X = nx1 size list of values of each node
        X = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        X[source] = [-1]
        X[destination] = [2]

        # Y = nodes to go to to reach destination. Minimum size: 1
        # path is reversed to follow s->d route
        Y = path[::-1]
        
        dataset.append([edge_index, X, Y])
    return dataset

# Generate a dataset
dataset = generate_dataset(10000, num_nodes)

# Create a DataFrame
df = pd.DataFrame(dataset, columns=["Edge index", "X", "Y"])

# Write the DataFrame to an CSV file
df.to_csv("data.csv", index=False)
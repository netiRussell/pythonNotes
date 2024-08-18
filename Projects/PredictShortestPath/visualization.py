from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

def visualize(dataset, num_nodes, run):
  if run == False:
    return

  # Dataset info
  data = dataset[0]

  # Visualization
  G = to_networkx(data, to_undirected=True)
  plt.figure(figsize=(7,7))
  plt.xticks([])
  plt.yticks([])
  nx.draw_networkx(G,
                  pos=nx.bfs_layout(G, 0),
                  with_labels=True)
  plt.show()
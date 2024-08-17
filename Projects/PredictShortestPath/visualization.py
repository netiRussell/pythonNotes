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
                  pos=nx.spring_layout(G, seed=42),
                  with_labels=False)
  nx.draw_networkx_labels(G,pos=nx.spring_layout(G, seed=42),labels={i: str(i) for i in range(num_nodes)},font_size=14,font_color='black')
  plt.show()
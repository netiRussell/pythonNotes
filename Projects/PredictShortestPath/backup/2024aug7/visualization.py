from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

def visualize(dataset, status):
  if status == False:
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
  nx.draw_networkx_labels(G,pos=nx.spring_layout(G, seed=42),labels={0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"11", 12:"12", 13:"13", 14:"14", 15:"15"},font_size=14,font_color='black')
  plt.show()
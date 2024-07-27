from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

def visualize(dataset, status):
  if status == False:
    return

  # Dataset info
  print(f'Number of features: {dataset.num_features}')
  print(f'Number of classes: {dataset.num_classes}')
  data = dataset[0]

  print(f'Number of nodes: {data.num_nodes}')
  print(f'Number of edges: {data.num_edges}')
  print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
  print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
  print(f'Contains self-loops: {data.has_self_loops()}')
  print(f'Is undirected: {data.is_undirected()}')

  # Visualization
  G = to_networkx(data, to_undirected=True)
  plt.figure(figsize=(7,7))
  plt.xticks([])
  plt.yticks([])
  nx.draw_networkx(G,
                  pos=nx.spring_layout(G, seed=42),
                  with_labels=False)
  nx.draw_networkx_labels(G,pos=nx.spring_layout(G, seed=42),labels={0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10",},font_size=14,font_color='black')
  plt.show()
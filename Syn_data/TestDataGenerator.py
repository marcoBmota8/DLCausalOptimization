# %%
import os 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from Gen_syn_data import DataGenerator

# %%
# Generate synthetic data
datagen = DataGenerator(
    p=10,
    num_irrelevant_dim=3,
    random_dim=2,
    n_branches=2
)

A = datagen.adjacency_matrix

# %%
# Create directed graph from adjacency matrix
G = nx.from_numpy_array(A, create_using=nx.DiGraph)

# Create a dictionary of labels
labels = {i: f'X{i+1}' for i in range(G.number_of_nodes())}

# Draw the graph
# Draw the nodes
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1000)

# Draw the labels
nx.draw_networkx_labels(G, pos, labels=labels, font_color='white')

# Draw the edges
nx.draw_networkx_edges(G, pos, node_size=1000, arrowstyle='-|>', arrowsize=20, width=2)

plt.show()

# %%
df, dag_info = datagen.generate_data(n=40, noise=0.005, threshold=0.5)
# %%

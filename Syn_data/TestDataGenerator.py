# %%
import os 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from Gen_syn_data import DataGenerator

# %%
# define DAG
A = np.array([
    [0,1,1,0],
    [0,0,1,1],
    [0,0,0,1],
    [0,0,0,0]
    ])

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
# Generate synthetic data
datagen = DataGenerator(
    adjacency_matrix=A,
    p=4,
    n=30,
    noise=1e-2,
    threshold=0.5
)
# %%
df = datagen.generate_data()
# %%

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

# Draw the graph
nx.draw(G, with_labels=True)
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

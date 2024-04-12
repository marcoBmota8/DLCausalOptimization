import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class DataGenerator:
    def __init__(self, p, num_irrelevant_dim, random_dim, n_branches):
        self.p = p # DAG dimensionality = relevant_dim + irrelevant_dim
        self.dim_irrev = num_irrelevant_dim # irrelevant number of nodes in the DAG
        self.dim_random = random_dim # random features
        self.n_branches = n_branches # number of branches to grow the DAG 
        self.adjanceny_matrix = self.generate_sparse_dag(p=self.p, n_branches=self.n_branches)

        # Define the unknown functions
        self.functions = [
            lambda x: x,  # Simple linear function
            self.mlp_tanh,  # MLP with tanh activation
            self.mlp_leaky_relu,  # MLP with LeakyReLU activation
            nn.LeakyReLU(),  # LeakyReLU
            lambda x: x**3,  # Polynomial of order three
            lambda x: np.log(x+1e-15), # logarithm
            torch.tanh  # Tanh
        ]

    def mlp_tanh(self, x):
        # Check if x is a PyTorch tensor, if not, convert it
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            
        model = nn.Sequential(
            nn.Linear(x.shape[1], 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
        return model(x)

    def mlp_leaky_relu(self, x):
        # Check if x is a PyTorch tensor, if not, convert it
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        model = nn.Sequential(
            nn.Linear(x.shape[1], 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        return model(x)

    def generate_data(self, n, noise, threshold):
        data = torch.from_numpy(np.random.normal(size=(n, self.p)) + noise).float()
        for i in range(self.p):
            parents = np.where(self.adjacency_matrix[:, i])[0]
            if parents.size > 0:
                fi = np.random.choice(self.functions)
                pa_data = fi(data[:, parents])
                # Non MLP functions need to aggregate the contribution of each parent via summation
                if pa_data.shape[1] > 1:
                    pa_data = pa_data.sum(axis=1).unsqueeze(dim=1)
                data[:, i] = pa_data.squeeze(dim=1)

        # Convert the target (node with no children) to binary
        binary_labels = (torch.sigmoid(data[:,self.sink]) > threshold).int()
        data[:, self.sink] = binary_labels

        # Generate strings for features names
        features_names = [f'X{i}' for i in range(1, self.p+1+self.dim_random)]
        features_names[self.sink] = 'T'
        # Generate the random (distratcion nodes)
        ...
                
        # convert to DataFrame
        df = pd.DataFrame(data.detach().numpy(), columns=features_names)
        
        # Shffle the columns 
        ...
        
        # Metadata info
        dag_info = {
            'adjacency_matrix': self.adjanceny_matrix,
            'target':self.sink,
            'relevant_nodes': self.sink_anc,
            'irrelevant_nodes': np.array(list(set(np.arange(self.p))-set(self.sink_anc)))
            
        }
        return data
    
    def find_ancestors(self, adjacency_matrix, node):
        visited = set()
        stack = [node]
        reference_node = node

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(np.where(adjacency_matrix[:, node])[0])
        # Remove the node itself if there is no self-loop
        if adjacency_matrix[reference_node, reference_node] == 0:
            visited.remove(reference_node)

        return list(visited)
    
    def generate_sparse_dag(self, p, n_branches):
        cycles = False
        while not cycles: # TODO check this can be avoided by ensuring all generated adjacency matrices are DAGs

            # Initialize an empty adjacency matrix
            adjacency_matrix = np.zeros((p, p))

            # Randomly select "n_branches" nodes to be the roots of the DAG branches
            roots = np.random.choice(p, n_branches, replace=False)

            # Add edges from the roots to other nodes
            for root in roots:
                children = np.random.choice([node for node in range(p) if node not in roots], size=np.random.randint(1, p//n_branches), replace=False)
                adjacency_matrix[root, children] = 1

            # Add edges from the other nodes to the sink node
            for node in range(p):
                if node not in roots and adjacency_matrix[:, node].sum() == 0:
                    children = np.random.choice([child for child in range(p) if child > node], size=np.random.randint(1, p//n_branches), replace=False)
                    adjacency_matrix[node, children] = 1

            # Make sure all nodes point to a single sink node
            sink = np.random.choice([node for node in range(p) if adjacency_matrix[node, :].sum() == 0])
            adjacency_matrix[(adjacency_matrix.sum(axis=1) > 0) & (adjacency_matrix.sum(axis=0) == 0), sink] = 1
            self.sink = sink # this will be the binary target
            self.sink_anc = self.find_ancestors(self.adjanceny_matrix,self.sink) # Ancestors of the sink node (target)
            
            # Add other sink nodes to the DAG that wont be ancestors of the target
            new_roots = np.random.choice(self.sink_anc, size = np.random.randint(1, len(self.sink_anc)), replace=False)
            for new_root in new_roots:
                valid_children = np.arange(p)[~np.isin(np.arange(p),np.append(self.sink,self.sink_anc)) & (np.arange(p) > new_root)] # nodes that are valid children for current new_root
                children = np.random.choice(valid_children, size=np.random.randint(1, len(valid_children)), replace=False)

            # check for cycles
            cycles = self.is_cyclic(adjacency_matrix) # TODO check this can be avoided by ensuring all generated adjacency matrices are DAGs

        return adjacency_matrix

    def is_cyclic_util(self, adjacency_matrix, node, visited, recursion_stack):
        # checks if a cycle exists starting from a given node.
        visited[node] = True
        recursion_stack[node] = True

        for child in np.where(adjacency_matrix[node, :])[0]:
            if visited[child] == False:
                if self.is_cyclic_util(adjacency_matrix, child, visited, recursion_stack) == True:
                    return True
            elif recursion_stack[child] == True:
                return True

        recursion_stack[node] = False
        return False
    
    def is_cyclic(self, adjacency_matrix):
        # Checks if adjacency matrix contains cycle(s)
        n = len(adjacency_matrix)
        visited = [False] * n
        recursion_stack = [False] * n
        for node in range(n):
            if visited[node] == False:
                if self.is_cyclic_util(adjacency_matrix, node, visited, recursion_stack) == True:
                    return True
        return False
    
    def generate_irrelevant_features(self):
        # TODO generate irrelevant features by drawing from gaussian and non-gaussian distributions and add them to the dataset
        return
        
        # TODO format output (data and ancestors indices) in generate_data function
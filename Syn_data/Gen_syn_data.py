import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class DataGenerator:
    def __init__(self, p, n, noise, threshold):
        self.p = p
        self.n = n
        self.noise = noise
        self.threshold = threshold
        
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

    def generate_data(self):
        data = torch.from_numpy(np.random.normal(size=(self.n, self.p)) + self.noise).float()
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
        target = np.where(~self.adjacency_matrix.any(axis=1))[0][0]
        binary_labels = (torch.sigmoid(data[:,target]) > self.threshold).int()
        data[:, target] = binary_labels

        # Convert to DataFrame
        data = pd.DataFrame(data.detach().numpy(), columns=[f'X{i}' for i in range(1, self.p+1)])

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
        while not cycles:
            # Initialize an empty adjacency matrix
            adjacency_matrix = np.zeros((p, p))

            # Randomly select "n_branches" nodes to be the roots of the DAG branches
            roots = np.random.choice(p, n_branches, replace=False)

            # Add edges from the roots to other nodes
            for root in roots:
                children = np.random.choice([node for node in range(p) if node not in roots], size=np.random.randint(1, p//2), replace=False)
                adjacency_matrix[root, children] = 1

            # Add edges from the other nodes to the sink node
            for node in range(p):
                if node not in roots and adjacency_matrix[:, node].sum() == 0:
                    children = np.random.choice([child for child in range(p) if child > node], size=np.random.randint(1, p//2), replace=False)
                    adjacency_matrix[node, children] = 1

            # Make sure all nodes point to a single sink node
            sink = np.random.choice([node for node in range(p) if adjacency_matrix[node, :].sum() == 0])
            adjacency_matrix[(adjacency_matrix.sum(axis=1) > 0) & (adjacency_matrix.sum(axis=0) == 0), sink] = 1
            
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
    
    def generate_irrelevant_features(self, k):
        # TODO generate irrelevant features by drawing from gaussian and non-gaussian distributions and add them to the dataset
        return
        
        # TODO format output (data and ancestors indices) in generate_data function
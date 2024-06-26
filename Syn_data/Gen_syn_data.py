import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class DataGenerator:
    def __init__(self, p, random_dim, n_branches):
        self.p = p # DAG dimensionality 
        self.dim_random = random_dim # random features
        self.n_branches = n_branches # number of branches to grow the DAG 
        self.adjacency_matrix, self.sink_anc, self.sink = self.generate_dag(p=self.p, n_branches=self.n_branches)

        # Define the unknown functions
        self.functions = [
            lambda x: x,  # Identity
            self.mlp_tanh,  # MLP with tanh activation
            self.mlp_leaky_relu,  # MLP with LeakyReLU activation
            nn.LeakyReLU(),  # LeakyReLU
            lambda x: x**3,  # Polynomial of order three
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
    
    def topological_sort(self, adjacency_matrix):
        # Number of nodes
        p = adjacency_matrix.shape[0]

        # Perform a topological sort
        visited = set()
        order = []
        for i in range(p):
            if i not in visited:
                stack = [i]
                while stack:
                    node = stack[-1]
                    if node not in visited:
                        visited.add(node)
                        for j in np.where(adjacency_matrix[node])[0]: # for children of node
                            if j not in visited:
                                stack.append(j)
                    else:
                        stack.pop()
                        order.append(node)
                        
        topo_order = order[::-1]

        return topo_order

    def generate_data(self, n, noise, threshold): # TODO: follow causal order of assignment.
        min_frac_classes = 0
        while (min_frac_classes>0.7) or (min_frac_classes<0.2): # Check that at least 20/70 of both classes
            order = self.topological_sort(self.adjacency_matrix)
            data = torch.from_numpy(np.random.normal(size=(n, self.p)) + np.random.normal(loc=0.0, scale=noise, size=(n,self.p))).float()
            for idx in order:
                parents = np.where(self.adjacency_matrix[:, idx])[0]
                if parents.size > 0:
                    fi = np.random.choice(self.functions)
                    pa_data = fi(data[:, parents])
                    # Non MLP functions need to aggregate the contribution of each parent via summation
                    if pa_data.shape[1] > 1:
                        pa_data = pa_data.sum(axis=1).unsqueeze(dim=1)
                    data[:, idx] = pa_data.squeeze(dim=1)

            # Convert the target (node with no children) to binary
            binary_labels = (torch.sigmoid(data[:,self.sink]) > threshold).int()
            min_frac_classes = binary_labels.float().mean()
            
        # Assign binary labels to the target node
        print('Fraction of 1s in the binary target: ',min_frac_classes)    
        data[:, self.sink] = binary_labels

        # Generate strings for features names
        features_names = [f'X{i}' for i in range(1, self.p+1+self.dim_random)]
        features_names[self.sink] = 'T'
        
        # Generate the random (distraction) nodes
        for rv in range(self.dim_random):
            data = torch.cat((data, self.generate_random_feature(n)), dim=1)
                
        # convert to DataFrame
        df = pd.DataFrame(data.detach().numpy(), columns=features_names)
        
        # Shuffle the columns 
        # Get the shuffled indices
        shuffled_indices = np.random.permutation(df.shape[1])

        # Shuffle the columns
        df = df[df.columns[shuffled_indices]]

        # Create the dictionary
        shuffle_dict = {original: shuffled for original, shuffled in zip(shuffled_indices, np.arange(df.shape[1]))}
        # Order dictionary by keys
        shuffle_dict = {k: shuffle_dict[k] for k in sorted(shuffle_dict)}
        
        # Metadata info
        dag_info = {
            'adjacency_matrix': self.adjacency_matrix,
            'target': shuffle_dict[self.sink],
            'relevant_nodes': [shuffle_dict[v] for v in self.sink_anc],
            'irrelevant_nodes': [shuffle_dict[v] for v in list(set(np.arange(self.p))-set(self.sink_anc))],
            'random_nodes': [shuffle_dict[v] for v in np.arange(self.p, self.p+self.dim_random)]
            }
        
        return df, dag_info
    
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
    
    def generate_dag(self, p, n_branches):
        cycles = True
        while cycles: # TODO check this can be avoided by ensuring all generated adjacency matrices are DAGs

            # Initialize an empty adjacency matrix
            adjacency_matrix = np.zeros((p, p))

            # Randomly select "n_branches" nodes to be the roots of the DAG branches
            roots = np.random.choice(p, n_branches, replace=False)

            # Add edges from the roots to other nodes
            for root in roots:
                children = np.random.choice([node for node in range(p) if node not in roots], size=np.random.randint(1, p//2), replace=False)
                adjacency_matrix[root, children] = 1

            # Create further chains involving more orphan nodes
            for node in range(p):
                if node not in roots and adjacency_matrix[:, node].sum() == 0: # Parentless nodes excluding roots
                    valid_children = np.arange(p)[np.arange(p) > root]  # TODO: can improve on this by doing DFS or BFS and find possible children among all nodes without incurring into cycles 
                    if valid_children.size==0:
                        pass
                    elif valid_children.size==1:
                         adjacency_matrix[node, valid_children.item()] = 1
                    else:
                        children = np.random.choice(valid_children, size=np.random.randint(1, len(valid_children)), replace=False)
                        adjacency_matrix[node, children] = 1

            # Make sure all terminal nodes (potential sink nodes) point to a single sink node (chosen one)
            sink = np.random.choice([node for node in range(p) if adjacency_matrix[node, :].sum() == 0], size = 1).item()
            adjacency_matrix[(adjacency_matrix.sum(axis=1) > 0) & (adjacency_matrix.sum(axis=0) == 0), sink] = 1 # Make connections between all parentless nodes with children and the target node to ensure it has at least n_branches ancestors.
            sink_anc = self.find_ancestors(adjacency_matrix,sink) # Ancestors of the sink node (target)
            
            # Add other sink nodes to the DAG that wont be ancestors of the target
            new_roots = np.random.choice(sink_anc, size = np.random.randint(1, len(sink_anc)), replace=False)
            for new_root in new_roots:
                valid_children = np.arange(p)[~np.isin(np.arange(p),np.append(sink,sink_anc)) & (np.arange(p) > new_root)] # nodes that are valid children for current new_root
                if valid_children.size==0:
                    pass
                else:
                    children = np.random.choice(valid_children, size=np.random.randint(1, len(valid_children)) if len(valid_children)>1 else 1, replace=False)
                    adjacency_matrix[new_root, children] = 1


            # check for cycles
            cycles = self.is_cyclic(adjacency_matrix) # TODO check this can be avoided by ensuring all generated adjacency matrices are DAGs

        return adjacency_matrix, sink_anc, sink

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
    
    def generate_random_feature(self, n):
        # List of distributions to choose from
        distributions = ['normal', 'uniform', 'poisson', 'binomial']

        # Randomly choose a distribution
        distribution = np.random.choice(distributions)

        # Draw n samples from the chosen distribution
        if distribution == 'normal':
            samples = np.random.normal(loc=0, scale=1, size=n)
        elif distribution == 'uniform':
            samples = np.random.uniform(low=0, high=1, size=n)
        elif distribution == 'poisson':
            samples = np.random.poisson(lam=1, size=n)
        elif distribution == 'binomial':
            samples = np.random.binomial(n=1, p=0.5, size=n)

        # Convert the samples to a (n, 1) tensor
        samples_tensor = torch.from_numpy(samples).view(n, 1)

            
        return samples_tensor

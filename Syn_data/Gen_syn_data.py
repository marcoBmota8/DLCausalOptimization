import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class DataGenerator:
    def __init__(self, adjacency_matrix, p, n, noise, threshold):
        self.adjacency_matrix = adjacency_matrix
        self.p = p
        self.n = n
        self.noise = noise
        self.threshold = threshold

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
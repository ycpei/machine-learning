import torch
from torch import nn

def fact(n):
    return np.prod(np.arange(1, n + 1))

class MLP(nn.Module):
    def __init__(input_size, hidden_size=1000):
        super().__init__()
        output_size = fact(input_size)
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.network(x)

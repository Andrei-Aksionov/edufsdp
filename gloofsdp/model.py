import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 18, hidden_dim: int = 24, output_dim: int = 12):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)

"""
Toy module definitions used for testing and demonstrating Fully Sharded Data Parallel (FSDP).

This simple Multi-Layer Perceptron (MLP) acts as a clean baseline to observe:
  1. Module tree traversal and parameter sharding (`layer1.weight`, `layer2.weight`, etc.)
  2. All-Gather collective calls before linear layers during forward/backward passes.
  3. Memory freeing (dropping full unsharded tensors) post-computation.

Data Flow & Architecture:

  Input [Batch, 18] ──> [Linear: 18 -> 24] ──> [ReLU] ──> [Linear: 24 -> 12] ──> Output [Batch, 12]
                            │                                  │
                   (Sharded across ranks)             (Sharded across ranks)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A simple 2-layer Multi-Layer Perceptron (MLP).

    This network is purposely kept small with distinct input/hidden/output dimensions
    to make parameter shape tracking easy when sharding across multi-rank nodes.
    Dimensions are selected to work with [2, 3, 4, 6] ranks.

    Args:
        input_dim (int): Feature dimension of input tensor. Defaults to 18.
        hidden_dim (int): Hidden representation dimension. Defaults to 24.
        output_dim (int): Dimension of output predictions. Defaults to 12.
    """

    def __init__(self, input_dim: int = 18, hidden_dim: int = 24, output_dim: int = 12) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass.

        Args:
            x (torch.Tensor): Input batch tensor of shape `(batch_size, input_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, output_dim)`.
        """
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)

"""Base test harness and model definitions for distributed unit tests."""

import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron module for testing sharding and hooks.

    Attributes:
        fc1: First linear projection without bias (in_features=16, out_features=4).
        fc2: Second linear projection without bias (in_features=4, out_features=8).
    """

    def __init__(self) -> None:
        super().__init__()
        # Weight shape: (out_features=4, in_features=16) -> Shard shape on 2 ranks: (2, 16)
        self.fc1 = nn.Linear(16, 4, bias=False)
        # Weight shape: (out_features=8, in_features=4) -> Shard shape on 2 ranks: (4, 4)
        self.fc2 = nn.Linear(4, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward pass through linear layers.

        Args:
            x: Input tensor of shape (batch_size, 16).

        Returns:
            Output tensor of shape (batch_size, 8).
        """
        return self.fc2(self.fc1(x))


class DistributedTestBase:
    """Base class for distributed unit tests executing under `torch.multiprocessing.spawn`."""

    @staticmethod
    def is_torch_mp_worker() -> bool:
        """Checks if current process is a worker spawned by the multi_rank marker runner."""
        return os.environ.get("PYTEST_TORCH_MP_SPAWN_WORKER") == "1"

    def setup_method(self, method: Any) -> None:
        """Sets up CPU device mesh and model instances inside worker process."""
        if not self.is_torch_mp_worker():
            return

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cpu")
        self.device_mesh = init_device_mesh("cpu", (self.world_size,))

        # Identical seed guarantees identical baseline parameters on every rank
        torch.manual_seed(42)
        self.base_model = SimpleMLP()

        # Target expected tensor shapes: parameter_name -> (global_shape, local_shard_shape)
        self.expected_shapes = {
            "fc1.weight": (
                torch.Size([4, 16]),
                torch.Size([4 // self.world_size, 16]),
            ),
            "fc2.weight": (
                torch.Size([8, 4]),
                torch.Size([8 // self.world_size, 4]),
            ),
        }

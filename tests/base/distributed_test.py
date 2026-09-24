import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh


class SimpleMLP(nn.Module):
    """Simple testing module with multi-dimensional linear layers."""

    def __init__(self) -> None:
        super().__init__()
        # fc1: (4, 16) -> Rank shard: (2, 16) on 2 ranks
        self.fc1 = nn.Linear(16, 4, bias=False)
        # fc2: (4, 8)  -> Rank shard: (2, 8) on 2 ranks
        self.fc2 = nn.Linear(4, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class DistributedTestBase:
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    @staticmethod
    def is_torch_mp_worker() -> bool:
        """Return whether the test is executing inside a torch.multiprocessing.spawn worker. The parent pytest process
        also runs pytest lifecycle methods such as ``setup_method``. This helper distinguishes that process from the
        child processes spawned by the ``multi_rank`` pytest marker, where distributed initialization is expected to
        occur."""
        return os.environ.get("PYTEST_TORCH_MP_SPAWN_WORKER") == "1"

    def setup_method(self, method) -> None:
        """Executes INSIDE each worker process immediately after mp.spawn."""
        if not self.is_torch_mp_worker():
            return  # Skip initialization on parent PyTest runner process

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=28),
        )

        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cpu")
        self.device_mesh = init_device_mesh("cpu", (self.world_size,))

        torch.manual_seed(42)
        self.base_model = SimpleMLP()

        # Target expected shapes: layer_name -> (global_shape, local_shard_shape)
        self.expected_shapes = {
            "fc1.weight": (torch.Size([4, 16]), torch.Size([4 // self.world_size, 16])),
            "fc2.weight": (torch.Size([8, 4]), torch.Size([8 // self.world_size, 4])),
        }

    def teardown_method(self, method) -> None:
        """Tears down process group after test completion."""
        if dist.is_initialized():
            dist.destroy_process_group()

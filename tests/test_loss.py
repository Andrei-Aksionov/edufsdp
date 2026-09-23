import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from gloofsdp.wrapper import EducationalFSDP


def init_distributed() -> tuple[int, int]:
    """Initializes distributed backend without overriding base seed."""
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=28),  # Raises RuntimeError on hanging collective
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


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


class TestFSDPLossDecrease:
    def setup_method(self, method) -> None:
        """Executes setup inside child worker processes spawned by mp.spawn."""
        if "RANK" not in os.environ:
            return  # Skip parent process setup

        self.rank, self.world_size = init_distributed()
        self.device = torch.device("cpu")
        self.mesh = init_device_mesh("cpu", mesh_shape=(self.world_size,))

        # Identical seed across all ranks to guarantee identical initial weights
        torch.manual_seed(42)
        self.base_model = SimpleMLP().to(self.device)

        # Per-rank seeding ONLY for training data generation
        torch.manual_seed(100 + self.rank)
        self.x = torch.randn(2, 16, device=self.device, requires_grad=True)
        self.target = torch.randn(2, 8, device=self.device)

    def teardown_method(self, method) -> None:
        """Tears down process group after test completion."""
        if dist.is_initialized():
            dist.destroy_process_group()

    @pytest.mark.multi_rank(world_size=2)
    def test_loss_decreases_across_ranks(self) -> None:
        """Executed inside each rank process via PyTest mp.spawn hook."""
        fsdp_model = EducationalFSDP(self.base_model, device_mesh=self.mesh)
        fsdp_optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        losses = []

        # 5-step training convergence loop
        for _ in range(5):
            fsdp_optimizer.zero_grad()
            fsdp_out = fsdp_model(self.x)
            fsdp_loss = criterion(fsdp_out, self.target)
            fsdp_loss.backward()
            fsdp_optimizer.step()

            losses.append(fsdp_loss.item())

        # Assert loss convergence directly inside each worker rank
        assert len(losses) > 1, f"[Rank {self.rank}] Recorded insufficient loss values."
        start_loss, end_loss = losses[0], losses[-1]

        assert end_loss < start_loss, (
            f"[Rank {self.rank}] Loss failed to decrease: initial={start_loss:.6f}, final={end_loss:.6f}"
        )

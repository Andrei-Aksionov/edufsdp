import pytest
import torch
from torch import nn

from gloofsdp.fsdp import FSDP
from tests.base.distributed_test import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestFSDPLossDecrease(DistributedTestBase):
    def test_loss_decreases_across_ranks(self) -> None:
        """Executed inside each rank process via PyTest mp.spawn hook."""
        fsdp_model = FSDP(self.base_model, device_mesh=self.device_mesh)
        fsdp_optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        losses = []

        # Per-rank seeding ONLY for training data generation
        torch.manual_seed(100 + self.rank)
        x = torch.randn(2, 16, device=self.device, requires_grad=True)
        target = torch.randn(2, 8, device=self.device)

        # 5-step training convergence loop
        for _ in range(5):
            fsdp_optimizer.zero_grad()
            fsdp_out = fsdp_model(x)
            fsdp_loss = criterion(fsdp_out, target)
            fsdp_loss.backward()
            fsdp_optimizer.step()

            losses.append(fsdp_loss.item())

        # Assert loss convergence directly inside each worker rank
        assert len(losses) > 1, f"[Rank {self.rank}] Recorded insufficient loss values."
        start_loss, end_loss = losses[0], losses[-1]

        assert end_loss < start_loss, (
            f"[Rank {self.rank}] Loss failed to decrease: initial={start_loss:.6f}, final={end_loss:.6f}"
        )

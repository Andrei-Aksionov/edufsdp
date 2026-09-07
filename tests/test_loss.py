import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from gloofsdp.wrapper import EducationalFSDP
from tests.conftest import get_free_port


def init_distributed():
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=28),  # Raises RuntimeError on hanging collective
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)
    return rank, world_size


class MLP(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)


class TestFSDPLossDecrease:
    RESULTS_FILENAME = "results.json"

    def _worker_entrypoint(self):
        """Executed by torchrun via: python -m torch.distributed.run -m path.to.test_file result_path"""
        result_path = sys.argv[2]

        rank, world_size = init_distributed()
        device = torch.device("cpu")
        mesh = init_device_mesh("cpu", mesh_shape=(dist.get_world_size(),))

        # 1. Instantiate reference model on Rank 0 and broadcast to ensure 100% weight alignment
        base_model = MLP().to(device)

        fsdp_model = EducationalFSDP(base_model, device_mesh=mesh)
        fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # 3. Create Deterministic Per-Rank Input Data & Target Data
        x = torch.randn(2, 8, device=device)
        target = torch.randn(2, 4, device=device)

        print(f"[Rank {rank}] Running 5-step convergence comparison...")

        local_losses = []

        # 4. Training Step Verification Loop
        for step in range(1, 6):
            # --- Educational FSDP Step ---
            fsdp_optimizer.zero_grad()
            fsdp_out = fsdp_model(x)
            fsdp_loss = criterion(fsdp_out, target)
            fsdp_loss.backward()
            fsdp_optimizer.step()

            local_losses.append(fsdp_loss.item())

        gathered_losses = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_losses, local_losses)

        if rank == 0:
            rank_loss_dict = dict(enumerate(gathered_losses))
            with open(result_path, "w") as f:
                json.dump(rank_loss_dict, f)

        dist.destroy_process_group()

    def test_loss_decreases_across_ranks(self, tmp_path):
        """Standard Pytest entrypoint (Parent process)."""
        result_path = str(tmp_path / self.RESULTS_FILENAME)

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={get_free_port()}",
            "-m",
            __name__,  # Executes the current module
            self.__class__.__name__,  # Passed to sys.argv[1]
            result_path,  # Passed to sys.argv[2]
        ]

        env = os.environ.copy()
        env["TORCH_CPP_LOG_LEVEL"] = "ERROR"

        try:
            result = subprocess.run(cmd, env=env, timeout=30, capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"torchrun worker failed:\n{result.stderr}\n{result.stdout}")
        except subprocess.TimeoutExpired:
            pytest.fail("torchrun execution timed out (potential deadlock).")

        assert os.path.exists(result_path), "Worker failed to generate loss results file."
        with open(result_path) as f:
            rank_loss_dict = json.load(f)

        for rank_str, losses in rank_loss_dict.items():
            assert len(losses) > 1, f"Rank {rank_str} recorded only one or no losses."
            start_loss, end_loss = losses[0], losses[-1]
            assert end_loss < start_loss, f"Rank {rank_str} loss failed to decrease: {end_loss} !< {start_loss}"


if __name__ == "__main__":
    class_name = sys.argv[1]
    globals()[class_name]()._worker_entrypoint()

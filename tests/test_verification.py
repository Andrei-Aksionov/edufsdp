import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as NativeFSDP

from gloofsdp.wrapper import EducationalFSDP


class ToyModel(nn.Module):
    """A multi-layer model with Linear layers and GELU activations."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 8, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def init_distributed():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)
    return rank, world_size


@pytest.mark.torchrun(nproc=2)
def test_run_verification_suite():
    rank, world_size = init_distributed()
    device = torch.device("cpu")
    mesh = init_device_mesh("cpu", mesh_shape=(dist.get_world_size(),))

    # 1. Instantiate reference model on Rank 0 and broadcast to ensure 100% weight alignment
    base_model = ToyModel().to(device)
    for param in base_model.parameters():
        dist.broadcast(param.data, src=0)

    # Make deepcopies so both FSDP implementations start from identical parameters
    edu_raw_model = copy.deepcopy(base_model)
    native_raw_model = copy.deepcopy(base_model)

    # 2. Wrap Models
    # Educational FSDP (gloofsdp)
    edu_model = EducationalFSDP(edu_raw_model, device_mesh=mesh)
    edu_optimizer = torch.optim.SGD(edu_model.parameters(), lr=0.01)

    # PyTorch Native FSDP
    native_model = NativeFSDP(native_raw_model, device_mesh=mesh, device_id=device)
    native_optimizer = torch.optim.SGD(native_model.parameters(), lr=0.01)

    # 3. Create Deterministic Per-Rank Input Data & Target Data
    # Fixed seed for inputs so run-to-run comparisons are identical
    torch.manual_seed(100 + rank)
    x = torch.randn(4, 16, device=device)
    target = torch.randn(4, 8, device=device)
    criterion = nn.MSELoss()

    print(f"[Rank {rank}] Running 5-step convergence comparison...")

    # 4. Training Step Verification Loop
    for step in range(1, 6):
        # --- Educational FSDP Step ---
        edu_optimizer.zero_grad()
        edu_out = edu_model(x)
        edu_loss = criterion(edu_out, target)
        edu_loss.backward()
        edu_optimizer.step()

        # --- Native PyTorch FSDP Step ---
        native_optimizer.zero_grad()
        native_out = native_model(x)
        native_loss = criterion(native_out, target)
        native_loss.backward()
        native_optimizer.step()

        # --- Assertions & Checks ---
        # A. Check Forward Pass Output Matching
        output_diff = torch.abs(edu_out - native_out).max().item()

        # B. Check Loss Value Matching
        loss_diff = abs(edu_loss.item() - native_loss.item())

        print(f"[Rank {rank}] Step {step} | Forward Max Diff: {output_diff:.2e} | Loss Diff: {loss_diff:.2e}")

        # Tolerances for FP32 eager operations
        assert output_diff < 1e-5, f"Forward output diverged at step {step}!"
        assert loss_diff < 1e-5, f"Loss value diverged at step {step}!"

    print(f"✅ [Rank {rank}] ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    dist.destroy_process_group()

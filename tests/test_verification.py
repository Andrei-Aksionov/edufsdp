import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as NativeFSDP

from gloofsdp.fsdp import FSDP


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


def init_distributed():
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=28),  # Raises RuntimeError on hanging collective
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)
    return rank, world_size


@pytest.mark.multi_rank(world_size=2)
def test_run_verification_suite():
    rank = dist.get_rank()
    device = torch.device("cpu")
    mesh = init_device_mesh("cpu", mesh_shape=(dist.get_world_size(),))

    # 1. Instantiate reference model on Rank 0 and broadcast to ensure 100% weight alignment
    base_model = MLP().to(device)
    for parameter in base_model.parameters():
        dist.broadcast(parameter.data, src=0)

    # Make deepcopies so both FSDP implementations start from identical parameters
    edu_raw_model = copy.deepcopy(base_model)
    native_raw_model = copy.deepcopy(base_model)

    # 2. Wrap Models
    # Educational FSDP (gloofsdp)
    edu_model = FSDP(edu_raw_model, device_mesh=mesh)
    edu_optimizer = torch.optim.SGD(edu_model.parameters(), lr=0.01)

    # PyTorch Native FSDP
    native_model = NativeFSDP(native_raw_model, device_mesh=mesh, device_id=device)
    native_optimizer = torch.optim.SGD(native_model.parameters(), lr=0.01)

    # 3. Create Deterministic Per-Rank Input Data & Target Data
    # Fixed seed for inputs so run-to-run comparisons are identical
    torch.manual_seed(100 + rank)
    x = torch.randn(2, 8, device=device, requires_grad=True)
    target = torch.randn(2, 4, device=device)
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


@pytest.mark.multi_rank(world_size=2)
def test_run_verification_loss_decreases():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cpu")
    mesh = init_device_mesh("cpu", mesh_shape=(dist.get_world_size(),))

    # 1. Instantiate reference model on Rank 0 and broadcast to ensure 100% weight alignment
    # base_model = ToyModel().to(device)
    base_model = MLP().to(device)

    fsdp_model = FSDP(base_model, device_mesh=mesh)
    fsdp_optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 3. Create Deterministic Per-Rank Input Data & Target Data
    # requires_grad=True is required for the input tensor because module-level backward
    # hooks (register_full_backward_pre_hook) on the first layer need an entry autograd node.
    # Without it, autograd prunes the input graph edge and emits a UserWarning when firing
    # backward hooks at the output node instead.
    x = torch.randn(2, 8, device=device, requires_grad=True)
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

    # 1. Gather all loss lists across ranks
    gathered_losses = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_losses, local_losses)

    # 2. Rank 0 runs assertions on all collected loss histories
    if rank == 0:
        rank_loss_dict = dict(enumerate(gathered_losses))

        for r, raw_losses in rank_loss_dict.items():
            initial_loss = raw_losses[0]
            final_loss = raw_losses[-1]

            assert final_loss < initial_loss, (
                f"Rank {r} loss failed to decrease. Start loss: {initial_loss:.4f}, End loss: {final_loss:.4f}"
            )

    print(f"✅ [Rank {rank}] Loss decreases successfully!")
    dist.destroy_process_group()

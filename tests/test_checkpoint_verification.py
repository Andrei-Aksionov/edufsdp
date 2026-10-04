import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import init_device_mesh
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as NativeFSDP

from gloofsdp.state_dict import StateDictType as EDUStateDictType

# Import your Educational FSDP implementation
from gloofsdp.wrapper import EducationalFSDP


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 8, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42)
    return rank, world_size


@pytest.mark.multi_rank(world_size=2)
def test_state_dict_compatibility():
    rank, world_size = init_distributed()
    device = torch.device("cpu")
    mesh = init_device_mesh("cpu", mesh_shape=(dist.get_world_size(),))

    # 1. Base Model Initialization & Weight Sync
    base_model = ToyModel().to(device)
    for param in base_model.parameters():
        dist.broadcast(param.data, src=0)

    # Wrap with EducationalFSDP and Native FSDP
    edu_model = EducationalFSDP(copy.deepcopy(base_model), device_mesh=mesh)
    native_model = NativeFSDP(copy.deepcopy(base_model), device_mesh=mesh, device_id=device)

    print(f"[Rank {rank}] Testing State Dict Extraction...")

    # ------------------------------------------------------------------
    # TEST A: FULL State Dict Parity (EducationalFSDP vs Native FSDP)
    # ------------------------------------------------------------------
    # Get Full State Dict from Educational FSDP (All-Gathers parameters into global shapes)
    edu_full_state_dict = edu_model.state_dict(state_dict_type=EDUStateDictType.FULL)

    # Get Full State Dict from Native PyTorch FSDP
    native_full_state_dict, _ = get_state_dict(
        native_model, optimizers=(), options=StateDictOptions(full_state_dict=True)
    )

    if rank == 0:
        assert set(edu_full_state_dict.keys()) == set(native_full_state_dict.keys()), (
            "State dict keys mismatch between EducationalFSDP and Native FSDP!"
        )

        for key in edu_full_state_dict:
            print(f"{key=}")
            edu_tensor = edu_full_state_dict[key]
            native_tensor = native_full_state_dict[key]

            # Verify shapes match full un-sharded parameter shapes
            assert edu_tensor.shape == native_tensor.shape, (
                f"Shape mismatch for {key}: Edu {edu_tensor.shape} vs Native {native_tensor.shape}"
            )

            # Verify values are identical
            max_diff = torch.abs(edu_tensor - native_tensor).max().item()
            assert max_diff < 1e-6, f"Value mismatch in full state_dict for {key}: max diff {max_diff}"

        print("  ✅ TEST A PASSED: Full state_dict is 100% identical to Native FSDP!")

    # ------------------------------------------------------------------
    # TEST B: Reloading FULL State Dict into Raw Non-FSDP Model
    # ------------------------------------------------------------------
    if rank == 0:
        raw_eval_model = ToyModel()
        # Load state dict produced by EducationalFSDP directly into standard PyTorch module
        missing_keys, unexpected_keys = raw_eval_model.load_state_dict(edu_full_state_dict)

        assert len(missing_keys) == 0 and len(unexpected_keys) == 0, (  # ruff: ignore[pytest-composite-assertion]
            "Failed to load EducationalFSDP full state_dict into standalone nn.Module!"
        )

        print("  ✅ TEST B PASSED: Saved FULL state_dict loads seamlessly into raw PyTorch module!")

    # ------------------------------------------------------------------
    # TEST C: Local / Sharded Checkpoint Resumption Cycle
    # ------------------------------------------------------------------
    # Simulate a training step to mutate weights
    x = torch.randn(4, 16, requires_grad=True)
    target = torch.randn(4, 8)
    criterion = nn.MSELoss()

    opt = torch.optim.SGD(edu_model.parameters(), lr=0.1)
    opt.zero_grad()
    loss = criterion(edu_model(x), target)
    loss.backward()
    opt.step()

    # Extract Local Sharded State Dict
    sharded_state_dict = edu_model.state_dict(EDUStateDictType.LOCAL)

    # Instantiate a fresh EducationalFSDP model and load the sharded state dict directly
    fresh_edu_model = EducationalFSDP(ToyModel().to(device), device_mesh=mesh)
    fresh_edu_model.load_state_dict(sharded_state_dict, EDUStateDictType.LOCAL)

    # Run forward pass on both original and restored models
    with torch.no_grad():
        out_orig = edu_model(x)
        out_restored = fresh_edu_model(x)
        restore_diff = torch.abs(out_orig - out_restored).max().item()

    assert restore_diff < 1e-6, f"Restored sharded model forward output diverged: diff {restore_diff}"
    print("  ✅ TEST C PASSED: Local sharded state_dict save/load cycle preserved exact model state!")

    if rank == 0:
        print("\n🎉 ALL CHECKPOINT & STATE_DICT COMPATIBILITY TESTS PASSED!")

    dist.destroy_process_group()

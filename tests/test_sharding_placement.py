"""
Unit tests verifying parameter placement and tensor sharding parity
between EducationalFSDP and native PyTorch FSDP.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from gloofsdp.wrapper import EducationalFSDP


class SimpleMLP(nn.Module):
    """Simple testing module with multi-dimensional linear layers."""

    def __init__(self) -> None:
        super().__init__()
        # Target weight shape: (4, 16) -> Expect rank shard: (2, 16) on 2 ranks
        self.fc1 = nn.Linear(16, 4, bias=False)
        self.fc2 = nn.Linear(4, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def setup_dist_group(rank: int, world_size: int) -> None:
    """Sets up a GLOO process group for CPU-based multi-rank testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )


def cleanup_dist_group() -> None:
    """Tears down distributed process group."""
    dist.destroy_process_group()


def run_rank_sharding_test(rank: int, world_size: int) -> None:
    """Execution body for each rank process."""
    setup_dist_group(rank, world_size)

    try:
        device = torch.device("cpu")
        device_mesh = init_device_mesh(str(device), (world_size,))

        # 1. Base Model Specs
        torch.manual_seed(42)
        model_custom = SimpleMLP()

        expected_global_shape = torch.Size([4, 16])
        expected_local_shard_shape = torch.Size([4 // world_size, 16])  # (2, 16)

        assert model_custom.fc1.weight.shape == expected_global_shape

        # 2. Wrap model with EducationalFSDP
        wrapped_custom = EducationalFSDP(model_custom, device_mesh=device_mesh)

        # 3. Static Parameter Sharding Checks (Rest State)
        custom_sharded_param = wrapped_custom.sharded_params["fc1.weight"]
        local_param = custom_sharded_param.local_param

        # Verify internal local shard shape
        assert local_param.shape == expected_local_shard_shape, (
            f"Rank {rank}: Custom local shard shape {local_param.shape} "
            f"does not match expected {expected_local_shard_shape}"
        )

        # Verify initial module attribute shape (Resting State = Sharded)
        assert wrapped_custom.module.fc1.weight.shape == expected_local_shard_shape

        # Verify element-based shard range indexing
        start_idx, end_idx = custom_sharded_param.shard_range
        shard_size = custom_sharded_param.global_shape.numel() // world_size
        expected_start = rank * shard_size
        expected_end = expected_start + shard_size
        assert (start_idx, end_idx) == (expected_start, expected_end)

        # 4. Dynamic Forward Execution Weight Shape Tracking
        captured_shapes = {}

        def pre_hook(module, inputs):
            # Captured right after EducationalFSDP ForwardPreHook gathers parameters
            captured_shapes["pre_forward"] = module.weight.shape

        def post_hook(module, inputs, output):
            # Captured right before EducationalFSDP ForwardPostHook frees parameters
            captured_shapes["during_forward"] = module.weight.shape

        # Register observation hooks on inner module
        wrapped_custom.module.fc1.register_forward_pre_hook(pre_hook)
        wrapped_custom.module.fc1.register_forward_hook(post_hook)

        # Execute single forward step
        dummy_input = torch.randn(2, 16)
        _ = wrapped_custom(dummy_input)

        # Check weight transition: Sharded -> Gathered (during forward) -> Sharded (post forward)
        assert captured_shapes["pre_forward"] == expected_global_shape, (
            f"Rank {rank}: Pre-forward gathered weight shape was "
            f"{captured_shapes['pre_forward']}, expected {expected_global_shape}"
        )
        assert wrapped_custom.module.fc1.weight.shape == expected_local_shard_shape, (
            f"Rank {rank}: Post-forward weight shape was not freed back to "
            f"{expected_local_shard_shape}, got {wrapped_custom.module.fc1.weight.shape}"
        )

        # 5. Native PyTorch FSDP Parity Check (Per-Submodule Sharding)
        torch.manual_seed(42)
        native_model = SimpleMLP()

        # Wrap submodules individually so native FSDP doesn't flatten fc1 and fc2 together
        fully_shard(native_model.fc1, mesh=device_mesh)
        fully_shard(native_model.fc2, mesh=device_mesh)
        fully_shard(native_model, mesh=device_mesh)

        # fc1.weight is now a DTensor sharded across rank 0/1
        fc1_native_param = native_model.fc1.weight.to_local().numel()
        expected_fc1_elements = expected_local_shard_shape.numel()  # 2 * 16 = 32 elements

        assert fc1_native_param == expected_fc1_elements, (
            f"Rank {rank}: Custom local elements ({local_param.numel()}) "
            f"does not match native FSDP fc1 elements ({fc1_native_param.numel()})"
        )

    finally:
        cleanup_dist_group()


@pytest.mark.parametrize("world_size", [2])
def test_parameter_sharding_and_rank_placement(world_size: int) -> None:
    """Spawns 2 process ranks to execute multi-node sharding checks."""
    torch.multiprocessing.spawn(
        run_rank_sharding_test,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

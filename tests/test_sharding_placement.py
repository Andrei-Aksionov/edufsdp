"""
Unit tests verifying parameter placement, lifecycle shape transitions,
and FSDP2 DTensor parity across all layers.
"""

import copy
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
        # fc1: (4, 16) -> Rank shard: (2, 16) on 2 ranks
        self.fc1 = nn.Linear(16, 4, bias=False)
        # fc2: (4, 8)  -> Rank shard: (2, 8) on 2 ranks
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


# class TestFSDPShardingAndPlacement:
class FSDPShardingAndPlacement:
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device_mesh = init_device_mesh("cpu", (world_size,))

        # Source base model instance
        torch.manual_seed(42)
        self.base_model = SimpleMLP()

        # Target expected shapes for all layers: layer_name -> (global_shape, local_shard_shape)
        self.expected_shapes = {
            "fc1.weight": (torch.Size([4, 16]), torch.Size([4 // world_size, 16])),
            # "fc2.weight": (torch.Size([4, 8]), torch.Size([4 // world_size, 8])),
            "fc2.weight": (torch.Size([8, 4]), torch.Size([8 // world_size, 4])),
        }

    def test_static_resting_sharding(self) -> None:
        """Verifies that all wrapped layer parameters rest in a sharded state matching rank placement."""
        model = copy.deepcopy(self.base_model)
        wrapped = EducationalFSDP(model, device_mesh=self.device_mesh)

        for param_name, (global_shape, expected_local_shape) in self.expected_shapes.items():
            sharded_param = wrapped.sharded_params[param_name]
            assert sharded_param.local_param.shape == expected_local_shape, (
                f"Rank {self.rank}: {param_name} internal local shape "
                f"{sharded_param.local_param.shape} != {expected_local_shape}"
            )
            assert sharded_param.global_shape == global_shape

            layer_name, attr = param_name.split(".")
            module_param = getattr(getattr(wrapped.module, layer_name), attr)
            assert module_param.shape == expected_local_shape, (
                f"Rank {self.rank}: {param_name} module resting shape {module_param.shape} != {expected_local_shape}"
            )

    def test_forward_pre_hook_all_gather_shapes(self) -> None:
        """Verifies that ForwardPreHook gathers all parameters to global shape prior to forward execution."""
        model = copy.deepcopy(self.base_model)
        wrapped = EducationalFSDP(model, device_mesh=self.device_mesh)

        gathered_shapes = {}

        for name, child in wrapped.module.named_children():

            def make_pre_hook(layer_key: str):
                def hook(module, inputs):
                    gathered_shapes[f"{layer_key}.weight"] = module.weight.shape

                return hook

            child.register_forward_pre_hook(make_pre_hook(name))

        dummy_input = torch.randn(2, 16)
        _ = wrapped(dummy_input)

        for param_name, (global_shape, _) in self.expected_shapes.items():
            assert gathered_shapes[param_name] == global_shape, (
                f"Rank {self.rank}: {param_name} gathered shape was "
                f"{gathered_shapes[param_name]}, expected {global_shape}"
            )

    def test_forward_post_hook_memory_cleanup_shapes(self) -> None:
        """Verifies that ForwardPostHook frees gathered parameters back to local sharded shapes post execution."""
        model = copy.deepcopy(self.base_model)
        wrapped = EducationalFSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(2, 16)
        _ = wrapped(dummy_input)

        for param_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = param_name.split(".")
            module_param = getattr(getattr(wrapped.module, layer_name), attr)
            assert module_param.shape == expected_local_shape, (
                f"Rank {self.rank}: {param_name} post-forward shape was not freed "
                f"back to {expected_local_shape}, got {module_param.shape}"
            )

    def test_fsdp2_dtensor_parity_all_layers(self) -> None:
        """Verifies element count parity against Native PyTorch FSDP2 (fully_shard) for all layers."""
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = EducationalFSDP(model_custom, device_mesh=self.device_mesh)

        model_native = copy.deepcopy(self.base_model)
        fully_shard(model_native.fc1, mesh=self.device_mesh)
        fully_shard(model_native.fc2, mesh=self.device_mesh)
        fully_shard(model_native, mesh=self.device_mesh)

        for param_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, _ = param_name.split(".")

            custom_local = wrapped_custom.sharded_params[param_name].local_param
            native_layer = getattr(model_native, layer_name)
            native_local = native_layer.weight.to_local()

            assert custom_local.shape == native_local.shape == expected_local_shape, (
                f"Rank {self.rank}: {param_name} shape mismatch - Custom {custom_local.shape}, "
                f"Native FSDP2 {native_local.shape}, Expected {expected_local_shape}"
            )


# --- Worker Launcher Helper ---


def _run_test_method(rank: int, world_size: int, test_method_name: str) -> None:
    """Spawns process group and dispatches execution to a single test method."""
    setup_dist_group(rank, world_size)
    try:
        # suite = TestFSDPShardingAndPlacement(rank, world_size)
        suite = FSDPShardingAndPlacement(rank, world_size)
        getattr(suite, test_method_name)()
    finally:
        cleanup_dist_group()


# --- Individual PyTest Test Entry Points ---


@pytest.mark.parametrize("world_size", [2])
def test_static_resting_sharding(world_size: int) -> None:
    torch.multiprocessing.spawn(
        _run_test_method,
        args=(world_size, "test_static_resting_sharding"),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2])
def test_forward_pre_hook_all_gather_shapes(world_size: int) -> None:
    torch.multiprocessing.spawn(
        _run_test_method,
        args=(world_size, "test_forward_pre_hook_all_gather_shapes"),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2])
def test_forward_post_hook_memory_cleanup_shapes(world_size: int) -> None:
    torch.multiprocessing.spawn(
        _run_test_method,
        args=(world_size, "test_forward_post_hook_memory_cleanup_shapes"),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2])
def test_fsdp2_dtensor_parity_all_layers(world_size: int) -> None:
    torch.multiprocessing.spawn(
        _run_test_method,
        args=(world_size, "test_fsdp2_dtensor_parity_all_layers"),
        nprocs=world_size,
        join=True,
    )

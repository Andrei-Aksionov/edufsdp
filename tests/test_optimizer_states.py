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


class OptimizerStatesTestSuite:
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cpu")
        self.device_mesh = init_device_mesh(str(self.device), (world_size,))

        # Source base model instance
        torch.manual_seed(42)
        self.base_model = SimpleMLP()

        # Target expected shapes for all layers: layer_name -> (global_shape, local_shard_shape)
        self.expected_shapes = {
            "fc1.weight": (torch.Size([4, 16]), torch.Size([4 // world_size, 16])),
            # "fc2.weight": (torch.Size([4, 8]), torch.Size([4 // world_size, 8])),
            "fc2.weight": (torch.Size([8, 4]), torch.Size([8 // world_size, 4])),
        }

    def test_optimizer_state_empty_before_step(self) -> None:
        """Verifies that optimizer states remain unitialized/empty across init, forward, and backward."""
        model = copy.deepcopy(self.base_model)
        wrapped = EducationalFSDP(model, device_mesh=self.device_mesh)
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=0.001)

        # 1. State should be empty right after instantiation
        assert len(optimizer.state) == 0, "Optimizer state should be empty after initialization"

        # 2. State should remain empty post-forward
        dummy_input = torch.randn(2, 16, device=self.device, requires_grad=True)
        output = wrapped(dummy_input)
        assert len(optimizer.state) == 0, "Optimizer state should be empty post-forward"

        # 3. State should remain empty post-backward
        loss = output.sum()
        loss.backward()
        assert len(optimizer.state) == 0, "Optimizer state should be empty post-backward (before step)"

    def test_optimizer_state_parity_post_step(self) -> None:
        """Verifies optimizer state lazy creation, local sharded shapes, and parity against FSDP2 after step()."""
        # 1. Setup Custom FSDP
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = EducationalFSDP(model_custom, device_mesh=self.device_mesh)
        optimizer_custom = torch.optim.Adam(wrapped_custom.parameters(), lr=0.001)

        # 2. Setup Native FSDP2
        model_native = copy.deepcopy(self.base_model)
        fully_shard(model_native.fc1, mesh=self.device_mesh)
        fully_shard(model_native.fc2, mesh=self.device_mesh)
        fully_shard(model_native, mesh=self.device_mesh)
        optimizer_native = torch.optim.Adam(model_native.parameters(), lr=0.001)

        # 3. Run Forward + Backward + Step with deterministic input
        dummy_input = torch.randn(2, 16, device=self.device, requires_grad=True)

        out_custom = wrapped_custom(dummy_input)
        out_custom.sum().backward()
        optimizer_custom.step()

        out_native = model_native(dummy_input)
        out_native.sum().backward()
        optimizer_native.step()

        # 4. Verify Custom FSDP state presence & shape parity against Native FSDP2
        for param_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = param_name.split(".")

            custom_param = getattr(getattr(wrapped_custom.module, layer_name), attr)
            native_param = getattr(getattr(model_native, layer_name), attr)

            # Ensure parameters are tracked inside optimizer state dicts
            assert custom_param in optimizer_custom.state, (
                f"Rank {self.rank}: {param_name} was not populated in custom optimizer.state"
            )
            assert native_param in optimizer_native.state, (
                f"Rank {self.rank}: {param_name} was not populated in native optimizer.state"
            )

            custom_state = optimizer_custom.state[custom_param]
            native_state = optimizer_native.state[native_param]

            # Verify Adam keys exist
            for state_key in ("exp_avg", "exp_avg_sq"):
                assert state_key in custom_state, (
                    f"Rank {self.rank}: Key '{state_key}' missing from custom optimizer state for {param_name}"
                )

                custom_tensor = custom_state[state_key]
                native_tensor = (
                    native_state[state_key].to_local()
                    if hasattr(native_state[state_key], "to_local")
                    else native_state[state_key]
                )

                # Assert shape matches expected local sharded shape and native FSDP shape
                assert custom_tensor.shape == native_tensor.shape == expected_local_shape, (
                    f"Rank {self.rank}: {param_name} state '{state_key}' shape mismatch - Custom "
                    f"{custom_tensor.shape}, Native {native_tensor.shape}, Expected {expected_local_shape}"
                )


# --- Worker Launcher Helper ---


def _run_suite_method(rank: int, world_size: int, test_method_name: str) -> None:
    """Spawns process group and dispatches execution to a single test method."""
    setup_dist_group(rank, world_size)
    try:
        suite = OptimizerStatesTestSuite(rank, world_size)
        getattr(suite, test_method_name)()
    finally:
        cleanup_dist_group()


# --- Generic Multiprocessing Test Launcher ---


def test_sharding_suite(test_method_name: str, world_size: int) -> None:
    """Single generic test entrypoint that executes any suite method across spawned ranks."""
    torch.multiprocessing.spawn(
        _run_suite_method,
        args=(world_size, test_method_name),
        nprocs=world_size,
        join=True,
    )


# --- PyTest Hook for Auto-Discovery ---


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrizes `test_sharding_suite` with every `test_*` method in `ShardingPlacementTestSuite`."""
    if "test_method_name" in metafunc.fixturenames:
        # Extract all method names starting with 'test_' from the container class
        method_names = [
            attr
            for attr in dir(OptimizerStatesTestSuite)
            if attr.startswith("test_") and callable(getattr(OptimizerStatesTestSuite, attr))
        ]
        metafunc.parametrize("test_method_name", method_names)
        metafunc.parametrize("world_size", [2])

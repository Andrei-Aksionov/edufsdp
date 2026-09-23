"""
Unit tests verifying parameter placement, lifecycle shape transitions,
and FSDP2 DTensor parity across all layers.
"""

import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

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


@pytest.mark.multi_rank(world_size=2)
class TestDTensorConsistency:
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    def setup_method(self, method) -> None:
        """Executes INSIDE each worker process immediately after mp.spawn."""
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
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

    def test_dtensor_type_lifecycle_forward(self) -> None:
        """Verifies that module parameters maintain DTensor instance types pre/post forward."""
        model = copy.deepcopy(self.base_model)
        wrapped = EducationalFSDP(model, device_mesh=self.device_mesh)

        # 1. Check resting state before forward
        for param_name in self.expected_shapes:
            layer_name, attr = param_name.split(".")
            module_param = getattr(getattr(wrapped.module, layer_name), attr)
            assert isinstance(module_param, DTensor), (
                f"Rank {self.rank}: Resting parameter {param_name} expected DTensor, got {type(module_param)}"
            )

        # 2. Run Forward Pass
        dummy_input = torch.randn(2, 16, device=self.device, requires_grad=True)
        _ = wrapped(dummy_input)

        # 3. Check resting state post-forward (should be resharded back to resting DTensor)
        for param_name in self.expected_shapes:
            layer_name, attr = param_name.split(".")
            module_param = getattr(getattr(wrapped.module, layer_name), attr)
            assert isinstance(module_param, DTensor), (
                f"Rank {self.rank}: Post-forward parameter {param_name} expected DTensor, got {type(module_param)}"
            )

    def test_dtensor_type_lifecycle_backward(self) -> None:
        """Verifies that module weights and .grad attributes are DTensor instances post-backward."""
        model = copy.deepcopy(self.base_model)
        wrapped = EducationalFSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(2, 16, device=self.device, requires_grad=True)
        output = wrapped(dummy_input)
        loss = output.sum()
        loss.backward()

        for param_name in self.expected_shapes:
            layer_name, attr = param_name.split(".")
            module_param = getattr(getattr(wrapped.module, layer_name), attr)

            # 1. Weight parameter type check
            assert isinstance(module_param, DTensor), (
                f"Rank {self.rank}: Post-backward {param_name} weight expected DTensor, got {type(module_param)}"
            )

            # 2. Gradient parameter type check
            assert module_param.grad is not None, f"Rank {self.rank}: {param_name}.grad is None"
            assert isinstance(module_param.grad, DTensor), (
                f"Rank {self.rank}: Post-backward {param_name}.grad expected DTensor, got {type(module_param.grad)}"
            )

    def test_dtensor_parity_against_native_fsdp2(self) -> None:
        """Verifies complete DTensor property parity (type, placement, mesh, grad type) against Native FSDP2."""
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = EducationalFSDP(model_custom, device_mesh=self.device_mesh)

        model_native = copy.deepcopy(self.base_model)
        fully_shard(model_native.fc1, mesh=self.device_mesh)
        fully_shard(model_native.fc2, mesh=self.device_mesh)
        fully_shard(model_native, mesh=self.device_mesh)

        dummy_input = torch.randn(2, 16, device=self.device, requires_grad=True)

        # Execute Forward + Backward cycle
        out_custom = wrapped_custom(dummy_input)
        out_custom.sum().backward()

        out_native = model_native(dummy_input)
        out_native.sum().backward()

        for param_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = param_name.split(".")

            custom_param = getattr(getattr(wrapped_custom.module, layer_name), attr)
            native_param = getattr(getattr(model_native, layer_name), attr)

            # 1. Parameter Type Parity (Both must be DTensors)
            assert isinstance(custom_param, DTensor), (
                f"Rank {self.rank}: Custom {param_name} expected DTensor, got {type(custom_param)}"
            )
            assert isinstance(native_param, DTensor), (
                f"Rank {self.rank}: Native {param_name} expected DTensor, got {type(native_param)}"
            )

            # 2. Placement & Mesh Parity
            assert custom_param.placements == native_param.placements, (
                f"Rank {self.rank}: {param_name} placement mismatch - Custom {custom_param.placements}, "
                f"Native {native_param.placements}"
            )
            assert custom_param.device_mesh == native_param.device_mesh, (
                f"Rank {self.rank}: {param_name} device_mesh mismatch - Custom {custom_param.device_mesh}, "
                f"Native {native_param.device_mesh}"
            )

            # 3. Local Shard Shape Parity
            assert custom_param.to_local().shape == native_param.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: {param_name} local shape mismatch - Custom {custom_param.to_local().shape}, "
                f"Native {native_param.to_local().shape}"
            )

            # 4. Gradient Type & Local Shape Parity
            assert isinstance(custom_param.grad, DTensor), (
                f"Rank {self.rank}: Custom {param_name}.grad expected DTensor, got {type(custom_param.grad)}"
            )
            assert isinstance(native_param.grad, DTensor), (
                f"Rank {self.rank}: Native {param_name}.grad expected DTensor, got {type(native_param.grad)}"
            )
            assert custom_param.grad.to_local().shape == native_param.grad.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: {param_name}.grad local shape mismatch - Custom {custom_param.grad.to_local().shape}, "
                f"Native {native_param.grad.to_local().shape}"
            )

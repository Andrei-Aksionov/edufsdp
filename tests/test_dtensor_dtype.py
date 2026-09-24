"""
Unit tests verifying parameter placement, lifecycle shape transitions,
and FSDP2 DTensor parity across all layers.
"""

import copy

import pytest
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from gloofsdp.wrapper import EducationalFSDP
from tests.base.distributed_test import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestDTensorConsistency(DistributedTestBase):
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

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

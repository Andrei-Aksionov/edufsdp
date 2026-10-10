"""
Unit tests verifying parameter placement, lifecycle shape transitions,
and FSDP2 DTensor parity across all layers.
"""

import copy

import pytest
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from gloofsdp.fsdp import FSDP
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestDTensorConsistency(DistributedTestBase):
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    def test_dtensor_type_lifecycle_forward(self) -> None:
        """Verifies that module parameters maintain DTensor instance types pre/post forward."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)

        # 1. Check resting state before forward
        for parameter_name in self.expected_shapes:
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped.inner_module, layer_name), attr)
            assert isinstance(module_parameter, DTensor), (
                f"Rank {self.rank}: Resting parameter {parameter_name} expected DTensor, got {type(module_parameter)}"
            )

        # 2. Run Forward Pass
        dummy_input = torch.randn(2, 16, device=self.device, requires_grad=True)
        _ = wrapped(dummy_input)

        # 3. Check resting state post-forward (should be resharded back to resting DTensor)
        for parameter_name in self.expected_shapes:
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped.inner_module, layer_name), attr)
            assert isinstance(module_parameter, DTensor), (
                f"Rank {self.rank}: Post-forward parameter {parameter_name} expected DTensor, got {type(module_parameter)}"
            )

    def test_dtensor_type_lifecycle_backward(self) -> None:
        """Verifies that module weights and .grad attributes are DTensor instances post-backward."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(2, 16, device=self.device, requires_grad=True)
        output = wrapped(dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name in self.expected_shapes:
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped.inner_module, layer_name), attr)

            # 1. Weight parameter type check
            assert isinstance(module_parameter, DTensor), (
                f"Rank {self.rank}: Post-backward {parameter_name} weight expected DTensor, got {type(module_parameter)}"
            )

            # 2. Gradient parameter type check
            assert module_parameter.grad is not None, f"Rank {self.rank}: {parameter_name}.grad is None"
            assert isinstance(module_parameter.grad, DTensor), (
                f"Rank {self.rank}: Post-backward {parameter_name}.grad expected DTensor, got {type(module_parameter.grad)}"
            )

    def test_dtensor_parity_against_native_fsdp2(self) -> None:
        """Verifies complete DTensor property parity (type, placement, mesh, grad type) against Native FSDP2."""
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = FSDP(model_custom, device_mesh=self.device_mesh)

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

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            native_parameter = getattr(getattr(model_native, layer_name), attr)

            # 1. Parameter Type Parity (Both must be DTensors)
            assert isinstance(custom_parameter, DTensor), (
                f"Rank {self.rank}: Custom {parameter_name} expected DTensor, got {type(custom_parameter)}"
            )
            assert isinstance(native_parameter, DTensor), (
                f"Rank {self.rank}: Native {parameter_name} expected DTensor, got {type(native_parameter)}"
            )

            # 2. Placement & Mesh Parity
            assert custom_parameter.placements == native_parameter.placements, (
                f"Rank {self.rank}: {parameter_name} placement mismatch - Custom {custom_parameter.placements}, "
                f"Native {native_parameter.placements}"
            )
            assert custom_parameter.device_mesh == native_parameter.device_mesh, (
                f"Rank {self.rank}: {parameter_name} device_mesh mismatch - Custom {custom_parameter.device_mesh}, "
                f"Native {native_parameter.device_mesh}"
            )

            # 3. Local Shard Shape Parity
            assert custom_parameter.to_local().shape == native_parameter.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} local shape mismatch - Custom {custom_parameter.to_local().shape}, "
                f"Native {native_parameter.to_local().shape}"
            )

            # 4. Gradient Type & Local Shape Parity
            assert isinstance(custom_parameter.grad, DTensor), (
                f"Rank {self.rank}: Custom {parameter_name}.grad expected DTensor, got {type(custom_parameter.grad)}"
            )
            assert isinstance(native_parameter.grad, DTensor), (
                f"Rank {self.rank}: Native {parameter_name}.grad expected DTensor, got {type(native_parameter.grad)}"
            )
            assert (
                custom_parameter.grad.to_local().shape == native_parameter.grad.to_local().shape == expected_local_shape
            ), (
                f"Rank {self.rank}: {parameter_name}.grad local shape mismatch - Custom {custom_parameter.grad.to_local().shape}, "
                f"Native {native_parameter.grad.to_local().shape}"
            )

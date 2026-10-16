"""
Integration tests verifying parameter DTensor types, placements,
sharded shapes, and lifecycle transitions.
"""

import copy

import pytest
import torch
from torch.distributed.tensor import DTensor, Shard

from gloofsdp.fsdp import FSDP
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestCustomDTensorLifecycle(DistributedTestBase):
    """
    Test suite covering static sharding, runtime weight DTensor placements,
    and lifecycle invariants.
    """

    def test_dtensor_resting_and_forward_lifecycle(self) -> None:
        """
        Verifies DTensor types, placements, and local shard shapes
        at rest and post-forward.
        """
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        # 1. Check resting state before forward
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped_model.inner_module, layer_name), attr)

            assert isinstance(module_parameter, DTensor)
            assert module_parameter.device_mesh == self.device_mesh
            assert module_parameter.placements == (Shard(0),)
            assert module_parameter.to_local().shape == expected_local_shape

        # 2. Run Forward Pass
        dummy_input = torch.randn(
            2, self.base_model.input_dim, device=self.device, requires_grad=True
        )
        _ = wrapped_model(dummy_input)

        # 3. Check resting state post-forward (should be resharded back to resting DTensor)
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped_model.inner_module, layer_name), attr)

            assert isinstance(module_parameter, DTensor)
            assert module_parameter.device_mesh == self.device_mesh
            assert module_parameter.placements == (Shard(0),)
            assert module_parameter.to_local().shape == expected_local_shape

    def test_dtensor_backward_lifecycle_and_gradients(self) -> None:
        """
        Verifies weight DTensors and gradient DTensor placements, shapes,
        and types post-backward.
        """
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(
            2, self.base_model.input_dim, device=self.device, requires_grad=True
        )
        output = wrapped_model(dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped_model.inner_module, layer_name), attr)

            # 1. Weight parameter checks (Type, Mesh, Placement, Sharded Shape)
            assert isinstance(module_parameter, DTensor)
            assert module_parameter.device_mesh == self.device_mesh
            assert module_parameter.placements == (Shard(0),)
            assert module_parameter.to_local().shape == expected_local_shape

            # 2. Gradient parameter checks (Type, Mesh, Placement, Local Shard Shape)
            assert module_parameter.grad is not None, (
                f"Rank {self.rank}: {parameter_name}.grad is None"
            )
            assert isinstance(module_parameter.grad, DTensor)
            assert module_parameter.grad.device_mesh == self.device_mesh
            assert module_parameter.grad.placements == (Shard(0),)
            assert module_parameter.grad.to_local().shape == expected_local_shape

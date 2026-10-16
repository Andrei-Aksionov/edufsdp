"""
Integration tests verifying parameter sharding, lifecycle shape transitions, and runtime
execution hook behaviors (all-gather and reduce-scatter memory cleanup) across layers.
"""

import copy

import pytest
import torch

from gloofsdp.fsdp import FSDP
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank
class TestShardingPlacement(DistributedTestBase):
    """
    Test suite covering static sharding, runtime weight lifecycle hooks, and sharded shape
    invariants.
    """

    def test_static_resting_sharding(self) -> None:
        """
        Verifies that all wrapped layer parameters rest in a sharded state matching rank
        placement.
        """
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(
                getattr(wrapped_model.inner_module, layer_name), attr
            ).to_local()
            assert module_parameter.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} module resting shape "
                f"{module_parameter.shape} != {expected_local_shape}"
            )

    def test_forward_pre_hook_all_gather_shapes(self) -> None:
        """
        Verifies that ForwardPreHook gathers all parameters to global shape prior to forward
        execution.
        """
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        gathered_shapes = {}

        for name, child in wrapped_model.inner_module.named_children():

            def make_pre_hook(layer_key: str):
                def hook(module, inputs):
                    gathered_shapes[f"{layer_key}.weight"] = module.weight.shape

                return hook

            child.register_forward_pre_hook(make_pre_hook(name))

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device)
        _ = wrapped_model(dummy_input)

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert gathered_shapes[parameter_name] == global_shape, (
                f"Rank {self.rank}: {parameter_name} gathered shape was "
                f"{gathered_shapes[parameter_name]}, expected {global_shape}"
            )

    def test_forward_post_hook_memory_cleanup_shapes(self) -> None:
        """
        Verifies that ForwardPostHook frees gathered parameters back to local sharded shapes
        post execution.
        """
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device)
        _ = wrapped_model(dummy_input)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(
                getattr(wrapped_model.inner_module, layer_name), attr
            ).to_local()
            assert module_parameter.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} post-forward shape was not freed "
                f"back to {expected_local_shape}, got {module_parameter.shape}"
            )

    def test_backward_pre_hook_all_gather_shapes(self) -> None:
        """Verifies that module weights are all-gathered during backward pass execution."""
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        pre_backward_parameter_shapes = {}

        for parameter_name, *_ in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            child_module = getattr(wrapped_model.inner_module, layer_name)

            def make_hook(full_key: str, mod: torch.nn.Module, attribute_name: str):
                def hook(module: torch.nn.Module, grad_output: tuple) -> None:
                    tensor_val = getattr(mod, attribute_name)
                    pre_backward_parameter_shapes[full_key] = tensor_val.shape

                return hook

            child_module.register_full_backward_pre_hook(
                make_hook(parameter_name, child_module, attr)
            )

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device)
        output = wrapped_model(dummy_input)
        loss = output.sum()
        loss.backward()

        assert len(pre_backward_parameter_shapes) == len(self.expected_shapes), (
            f"Rank {self.rank}: Captured {len(pre_backward_parameter_shapes)} parameters, "
            f"expected {len(self.expected_shapes)}"
        )

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert pre_backward_parameter_shapes[parameter_name] == global_shape, (
                f"Rank {self.rank}: {parameter_name} shape during backward was "
                f"{pre_backward_parameter_shapes[parameter_name]}, "
                f"expected global shape {global_shape}"
            )

    def test_backward_gradients_reduce_scattered(self) -> None:
        """
        Verifies that gradients are computed, sharded, and match local parameter shapes post
        loss.backward().
        """
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        torch.manual_seed(42)
        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device)
        output = wrapped_model(dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            parameter = getattr(getattr(wrapped_model.inner_module, layer_name), attr)
            local_grad = parameter.grad.to_local()

            assert local_grad is not None, (
                f"Rank {self.rank}: {parameter_name} grad was None after loss.backward()"
            )

            grad_shape = local_grad.shape
            assert grad_shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} local grad shape {grad_shape} "
                f"does not match expected local parameter shape {expected_local_shape}"
            )

    def test_backward_post_hook_memory_cleanup_and_sharded_grads(self) -> None:
        """
        Verifies that module weights return to sharded shapes post-backward and
        module.weight.grad holds sharded gradients.
        """
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device)
        output = wrapped_model(dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped_model.inner_module, layer_name), attr)

            # 1. Weight memory must be freed back to local sharded shape on the actual module
            # attribute
            assert module_parameter.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name} weight shape was not freed "
                f"back to {expected_local_shape}, got {module_parameter.to_local().shape}"
            )

            # 2. Gradient must exist directly on the module parameter attribute and match the
            # sharded shape
            assert module_parameter.grad.to_local() is not None, (
                f"Rank {self.rank}: {parameter_name}.grad was None on "
                "module parameter post-backward"
            )
            assert module_parameter.grad.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name}.grad shape was "
                f"{module_parameter.grad.to_local().shape}, "
                f"expected sharded shape {expected_local_shape}"
            )

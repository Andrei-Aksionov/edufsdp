"""
Unit tests verifying parameter placement, lifecycle shape transitions,
and FSDP2 DTensor parity across all layers.
"""

import copy

import pytest
import torch
from torch.distributed.fsdp import fully_shard

from gloofsdp.fsdp import FSDP
from tests.base.distributed_test import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestShardingPlacement(DistributedTestBase):
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    def test_static_resting_sharding(self) -> None:
        """Verifies that all wrapped layer parameters rest in a sharded state matching rank placement."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped.inner_module, layer_name), attr).to_local()
            assert module_parameter.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} module resting shape {module_parameter.shape} != {expected_local_shape}"
            )

    def test_forward_pre_hook_all_gather_shapes(self) -> None:
        """Verifies that ForwardPreHook gathers all parameters to global shape prior to forward execution."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)

        gathered_shapes = {}

        for name, child in wrapped.inner_module.named_children():

            def make_pre_hook(layer_key: str):
                def hook(module, inputs):
                    gathered_shapes[f"{layer_key}.weight"] = module.weight.shape

                return hook

            child.register_forward_pre_hook(make_pre_hook(name))

        dummy_input = torch.randn(2, 16)
        _ = wrapped(dummy_input)

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert gathered_shapes[parameter_name] == global_shape, (
                f"Rank {self.rank}: {parameter_name} gathered shape was "
                f"{gathered_shapes[parameter_name]}, expected {global_shape}"
            )

    def test_forward_post_hook_memory_cleanup_shapes(self) -> None:
        """Verifies that ForwardPostHook frees gathered parameters back to local sharded shapes post execution."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(2, 16)
        _ = wrapped(dummy_input)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped.inner_module, layer_name), attr).to_local()
            assert module_parameter.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} post-forward shape was not freed "
                f"back to {expected_local_shape}, got {module_parameter.shape}"
            )

    def test_fsdp2_dtensor_parity_all_layers(self) -> None:
        """Verifies element count parity against Native PyTorch FSDP2 (fully_shard) for all layers."""
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = FSDP(model_custom, device_mesh=self.device_mesh)

        model_native = copy.deepcopy(self.base_model)
        fully_shard(model_native.fc1, mesh=self.device_mesh)
        fully_shard(model_native.fc2, mesh=self.device_mesh)
        fully_shard(model_native, mesh=self.device_mesh)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_local = getattr(getattr(wrapped_custom.inner_module, layer_name), attr).to_local()
            native_local = getattr(getattr(model_native, layer_name), attr).to_local()

            assert custom_local.shape == native_local.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} shape mismatch - Custom {custom_local.shape}, "
                f"Native FSDP2 {native_local.shape}, Expected {expected_local_shape}"
            )

    def test_fsdp2_dtensor_parity_post_forward(self) -> None:
        """Verifies weight values, output parity, and None grads post-forward pass against FSDP2."""
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = FSDP(model_custom, device_mesh=self.device_mesh)

        model_native = copy.deepcopy(self.base_model)
        fully_shard(model_native.fc1, mesh=self.device_mesh)
        fully_shard(model_native.fc2, mesh=self.device_mesh)
        fully_shard(model_native, mesh=self.device_mesh)

        dummy_input = torch.randn(2, 16, device=self.device)

        # Execute forward pass on both models
        out_custom = wrapped_custom(dummy_input)
        out_native = model_native(dummy_input)

        # 1. Output parity
        torch.testing.assert_close(
            out_custom,
            out_native,
            msg=f"Rank {self.rank}: Forward output mismatch between Custom FSDP and Native FSDP2",
        )

        # 2. Weight shape, value parity, and grad checks post-forward
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(wrapped_custom.inner_module, layer_name), attr).to_local()
            native_parameter = getattr(getattr(model_native, layer_name), attr).to_local()

            # Shape parity
            assert custom_parameter.shape == native_parameter.shape == expected_local_shape, (
                f"Rank {self.rank}: Post-forward {parameter_name} shape mismatch - Custom {custom_parameter.shape}, "
                f"Native FSDP2 {native_parameter.shape}"
            )

            # Value parity (rested sharded weights unchanged)
            torch.testing.assert_close(
                custom_parameter,
                native_parameter,
                msg=f"Rank {self.rank}: Post-forward {parameter_name} weight values diverged from Native FSDP2",
            )

            # Gradients must be None post-forward
            assert custom_parameter.grad is None, (
                f"Rank {self.rank}: Expected {parameter_name}.grad to be None on Custom FSDP post-forward"
            )
            assert native_parameter.grad is None, (
                f"Rank {self.rank}: Expected {parameter_name}.grad to be None on Native FSDP2 post-forward"
            )

    # ---------- Backward step -------------------

    def test_backward_pass_sharded_gradients(self) -> None:
        """Verifies that gradients are computed, sharded, and match local parameter shapes post loss.backward()."""

        # 1. Custom EducationalFSDP execution
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = FSDP(model_custom, device_mesh=self.device_mesh)

        torch.manual_seed(42)
        dummy_input = torch.randn(2, 16)
        output_custom = wrapped_custom(dummy_input)
        loss_custom = output_custom.sum()
        loss_custom.backward()

        # 2. Native FSDP2 execution
        model_native = copy.deepcopy(self.base_model)
        fully_shard(model_native.fc1, mesh=self.device_mesh)
        fully_shard(model_native.fc2, mesh=self.device_mesh)
        fully_shard(model_native, mesh=self.device_mesh)

        torch.manual_seed(42)
        output_native = model_native(dummy_input)
        loss_native = output_native.sum()
        loss_native.backward()

        # 3. Assert local gradient shapes and FSDP2 parity for all sharded layers
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            # sharded_parameter = wrapped_custom.sharded_params[parameter_name]
            custom_parameter = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            custom_local_grad = custom_parameter.grad.to_local()

            # Verify local_parameter has a gradient populated
            assert custom_local_grad is not None, (
                f"Rank {self.rank}: {parameter_name} local_parameter.grad was None after loss.backward()"
            )

            # Verify gradient shape matches local parameter shape exactly
            custom_grad_shape = custom_local_grad.shape
            assert custom_grad_shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} local grad shape {custom_grad_shape} "
                f"does not match expected local parameter shape {expected_local_shape}"
            )

            # Verify local gradient element count parity against Native FSDP2
            native_parameter = getattr(getattr(model_native, layer_name), attr)
            native_local_grad = native_parameter.grad.to_local()

            assert custom_local_grad.shape == native_local_grad.shape, (
                f"Rank {self.rank}: {parameter_name} gradient element shape ({custom_local_grad.shape}) "
                f"does not match Native FSDP2 local gradient element shape ({native_local_grad.shape})"
            )

            try:
                torch.testing.assert_close(
                    custom_local_grad,
                    native_local_grad,
                    rtol=1e-5,
                    atol=1e-8,
                )
            except AssertionError as e:
                raise AssertionError(
                    f"Rank {self.rank}: {parameter_name} gradient mismatch against native FSDP2:\n{e}"
                ) from None

    def test_pre_backward_all_gather_shapes(self) -> None:
        """Verifies that module weights are all-gathered during backward pass execution."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)

        pre_backward_parameter_shapes = {}

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            child_module = getattr(wrapped.inner_module, layer_name)

            def make_hook(full_key: str, mod: torch.nn.Module, attribute_name: str):
                def hook(module: torch.nn.Module, grad_output: tuple) -> None:
                    # Direct attribute lookup gets full_tensor set by EducationalFSDP's BackwardPreHook
                    tensor_val = getattr(mod, attribute_name)
                    pre_backward_parameter_shapes[full_key] = tensor_val.shape

                return hook

            child_module.register_full_backward_pre_hook(make_hook(parameter_name, child_module, attr))

        dummy_input = torch.randn(2, 16)
        output = wrapped(dummy_input)
        loss = output.sum()
        loss.backward()

        # Assert that all expected parameters were captured and were at global shape during backward
        assert len(pre_backward_parameter_shapes) == len(self.expected_shapes), (
            f"Rank {self.rank}: Captured {len(pre_backward_parameter_shapes)} parameters, "
            f"expected {len(self.expected_shapes)}"
        )

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert pre_backward_parameter_shapes[parameter_name] == global_shape, (
                f"Rank {self.rank}: {parameter_name} shape during backward was "
                f"{pre_backward_parameter_shapes[parameter_name]}, expected global shape {global_shape}"
            )

    def test_post_backward_memory_cleanup_and_sharded_grads(self) -> None:
        """Verifies that module weights return to sharded shapes post-backward and module.weight.grad holds sharded gradients."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)

        dummy_input = torch.randn(2, 16)
        output = wrapped(dummy_input)
        loss = output.sum()
        loss.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            module_parameter = getattr(getattr(wrapped.inner_module, layer_name), attr)

            # 1. Weight memory must be freed back to local sharded shape on the actual module attribute
            assert module_parameter.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name} weight shape was not freed "
                f"back to {expected_local_shape}, got {module_parameter.to_local().shape}"
            )

            # 2. Gradient must exist directly on the module parameter attribute and match the sharded shape
            assert module_parameter.grad.to_local() is not None, (
                f"Rank {self.rank}: {parameter_name}.grad was None on module parameter post-backward"
            )
            assert module_parameter.grad.to_local().shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name}.grad shape was "
                f"{module_parameter.grad.to_local().shape}, expected sharded shape {expected_local_shape}"
            )

    def test_fsdp2_dtensor_parity_post_backward(self) -> None:
        """Verifies weight, gradient shape, and gradient value parity post-backward against FSDP2."""
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = FSDP(model_custom, device_mesh=self.device_mesh)

        model_native = copy.deepcopy(self.base_model)
        fully_shard(model_native.fc1, mesh=self.device_mesh)
        fully_shard(model_native.fc2, mesh=self.device_mesh)
        fully_shard(model_native, mesh=self.device_mesh)

        # Use identical input to ensure identical gradient computations
        dummy_input = torch.randn(2, 16, device=self.device)

        out_custom = wrapped_custom(dummy_input)
        loss_custom = out_custom.sum()
        loss_custom.backward()

        out_native = model_native(dummy_input)
        loss_native = out_native.sum()
        loss_native.backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            native_parameter = getattr(getattr(model_native, layer_name), attr)

            # 1. Weight shape & value parity post-backward (restored sharded state)
            custom_weight_local = custom_parameter.to_local()
            native_weight_local = native_parameter.to_local()

            assert custom_weight_local.shape == native_weight_local.shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name} weight shape mismatch - Custom "
                f"{custom_weight_local.shape}, Native FSDP2 {native_weight_local.shape}"
            )

            torch.testing.assert_close(
                custom_weight_local,
                native_weight_local,
                msg=f"Rank {self.rank}: Post-backward {parameter_name} weight values diverged from Native FSDP2",
            )

            # 2. Gradient existence & shape checks
            assert custom_parameter.grad is not None, (
                f"Rank {self.rank}: {parameter_name}.grad was None on Custom FSDP post-backward"
            )
            assert native_parameter.grad is not None, (
                f"Rank {self.rank}: {parameter_name}.grad was None on Native FSDP2 post-backward"
            )

            # For Native FSDP2 DTensors, grad is also a DTensor/tensor whose local shard matches sharded_shape
            custom_grad_local = (
                custom_parameter.grad.to_local()
                if hasattr(custom_parameter.grad, "to_local")
                else custom_parameter.grad
            )
            native_grad_local = (
                native_parameter.grad.to_local()
                if hasattr(native_parameter.grad, "to_local")
                else native_parameter.grad
            )

            assert custom_grad_local.shape == native_grad_local.shape == expected_local_shape, (
                f"Rank {self.rank}: Post-backward {parameter_name}.grad shape mismatch - Custom "
                f"{custom_grad_local.shape}, Native FSDP2 {native_grad_local.shape}"
            )

            # 3. Gradient value parity (Reduce-Scatter numerical parity)
            torch.testing.assert_close(
                custom_grad_local,
                native_grad_local,
                msg=f"Rank {self.rank}: Post-backward {parameter_name}.grad values diverged from Native FSDP2",
            )

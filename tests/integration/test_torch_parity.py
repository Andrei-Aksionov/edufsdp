"""Integration parity tests comparing edufsdp against PyTorch native FSDP v2 (fully_shard)."""

import copy
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from edufsdp.fsdp import FSDP
from edufsdp.state_dict import StateDictType
from tests.base.test_class import DistributedTestBase


def _reference_fully_shard_model(model: nn.Module, device_mesh) -> nn.Module:
    """Recursively discovers and fully_shards all submodules owning parameters dynamically."""
    for _, submodule in model.named_modules():
        if any(p.requires_grad for p in submodule.parameters(recurse=False)):
            fully_shard(submodule, mesh=device_mesh)
    fully_shard(model, mesh=device_mesh)
    return model


@pytest.mark.multi_rank
class TestStateDictParity(DistributedTestBase):
    """
    Verifies state dictionary extraction parity (Local and Full) between EducationalFSDP and
    Native FSDP2.
    """

    def test_full_state_dict_parity(self) -> None:
        """Verifies full (unsharded) state dict parity matches PyTorch native FSDP2."""
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        custom_sd = wrapped_custom.state_dict(state_dict_type=StateDictType.Full)
        reference_sd, _ = get_state_dict(
            wrapped_reference, optimizers=(), options=StateDictOptions(full_state_dict=True)
        )

        if self.rank == 0:
            assert set(custom_sd.keys()) == set(reference_sd.keys()), (
                "State dict keys mismatch between EducationalFSDP and Native FSDP!"
            )
            for key in custom_sd:
                torch.testing.assert_close(
                    custom_sd[key], reference_sd[key], rtol=1e-6, atol=1e-6
                )

    def test_local_state_dict_parity(self) -> None:
        """Verifies local (sharded) state dict parity matches PyTorch native FSDP2."""
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        custom_sd = wrapped_custom.state_dict(state_dict_type=StateDictType.Local)
        reference_sd, _ = get_state_dict(
            wrapped_reference, optimizers=(), options=StateDictOptions(full_state_dict=False)
        )

        assert set(custom_sd.keys()) == set(reference_sd.keys()), (
            "Local state dict keys mismatch between EducationalFSDP and Native FSDP!"
        )
        for key in custom_sd:
            custom_value = custom_sd[key]
            reference_value = reference_sd[key]
            assert isinstance(custom_value, DTensor)
            assert isinstance(reference_value, DTensor)

            torch.testing.assert_close(
                custom_value.to_local(), reference_value.to_local(), rtol=1e-6, atol=1e-6
            )


@pytest.mark.multi_rank
class TestDTensorStructureParity(DistributedTestBase):
    """
    Verifies DTensor property, placement, mesh, and gradient structure parity against
    FSDP2.
    """

    def test_dtensor_properties_and_gradients_parity(self) -> None:
        """
        Verifies parameter types, placements, device mesh, local shapes, and gradient
        DTensor types.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        dummy_input = torch.randn(
            2, self.base_model.input_dim, device=self.device, requires_grad=True
        )

        out_custom = wrapped_custom(dummy_input)
        out_custom.sum().backward()

        out_reference = wrapped_reference(dummy_input)
        out_reference.sum().backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            reference_parameter = getattr(getattr(wrapped_reference, layer_name), attr)

            assert isinstance(custom_parameter, DTensor)
            assert isinstance(reference_parameter, DTensor)
            assert custom_parameter.placements == reference_parameter.placements
            assert custom_parameter.device_mesh == reference_parameter.device_mesh
            assert (
                custom_parameter.to_local().shape
                == reference_parameter.to_local().shape
                == expected_local_shape
            )

            assert isinstance(custom_parameter.grad, DTensor)
            assert isinstance(reference_parameter.grad, DTensor)
            assert (
                custom_parameter.grad.to_local().shape
                == reference_parameter.grad.to_local().shape
                == expected_local_shape
            )


@pytest.mark.multi_rank
class TestOptimizerStateParity(DistributedTestBase):
    """Verifies optimizer state sharding and structure parity against FSDP2."""

    def test_optimizer_state_sharding_parity(self) -> None:
        """
        Verifies optimizer state buffer names, sharding shapes, and parity post optimizer
        step().
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        optimizer_custom = torch.optim.Adam(wrapped_custom.parameters(), lr=0.001)

        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )
        optimizer_reference = torch.optim.Adam(wrapped_reference.parameters(), lr=0.001)

        dummy_input = torch.randn(
            2, self.base_model.input_dim, device=self.device, requires_grad=True
        )

        out_custom = wrapped_custom(dummy_input)
        out_custom.sum().backward()
        optimizer_custom.step()

        out_reference = wrapped_reference(dummy_input)
        out_reference.sum().backward()
        optimizer_reference.step()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            reference_parameter = getattr(getattr(wrapped_reference, layer_name), attr)

            assert custom_parameter in optimizer_custom.state
            assert reference_parameter in optimizer_reference.state

            custom_state = optimizer_custom.state[custom_parameter]
            reference_state = optimizer_reference.state[reference_parameter]

            for state_key in ("exp_avg", "exp_avg_sq"):
                assert state_key in custom_state
                custom_tensor = (
                    custom_state[state_key].to_local()
                    if hasattr(custom_state[state_key], "to_local")
                    else custom_state[state_key]
                )
                reference_tensor = (
                    reference_state[state_key].to_local()
                    if hasattr(reference_state[state_key], "to_local")
                    else reference_state[state_key]
                )

                assert custom_tensor.shape == reference_tensor.shape == expected_local_shape


@pytest.mark.multi_rank
class TestExecutionLifecycleParity(DistributedTestBase):
    """
    Verifies strict parity of execution lifecycle state transitions between EducationalFSDP
    and Native FSDP2.
    """

    def test_sharding_after_wrapping_parity(self) -> None:
        """
        1. Verifies that parameters are correctly sharded immediately upon wrapping for both
        models.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        for parameter_name, (
            _global_shape,
            expected_local_shape,
        ) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            reference_param = getattr(getattr(wrapped_reference, layer_name), attr)

            # Both must be sharded to local shape initially
            assert custom_param.to_local().shape == expected_local_shape
            assert reference_param.to_local().shape == expected_local_shape
            assert not wrapped_custom.sharded_parameters[parameter_name].is_gathered

    def test_forward_parameters_are_gathered_parity(self) -> None:
        """
        2. Verifies that parameters are all-gathered during the forward pass in both
        models.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        custom_shapes_during_forward: dict[str, Size] = {}
        reference_shapes_during_forward: dict[str, Size] = {}

        def make_hook(model: nn.Module, shapes_dict: dict[str, Size]):
            def hook(module: nn.Module, _args: Any):
                for parameter_name in self.expected_shapes:
                    layer_name, attr = parameter_name.split(".")
                    if module == getattr(model, layer_name):
                        param = getattr(module, attr)
                        shapes_dict[parameter_name] = param.shape

            return hook

        custom_handles = [
            getattr(wrapped_custom.inner_module, p.split(".")[0]).register_forward_pre_hook(
                make_hook(
                    model=wrapped_custom.inner_module,
                    shapes_dict=custom_shapes_during_forward,
                )
            )
            for p in self.expected_shapes
        ]

        reference_handles = [
            getattr(wrapped_reference, p.split(".")[0]).register_forward_pre_hook(
                make_hook(
                    model=wrapped_reference,
                    shapes_dict=reference_shapes_during_forward,
                )
            )
            for p in self.expected_shapes
        ]

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device)
        wrapped_custom(dummy_input)
        wrapped_reference(dummy_input)

        for h in custom_handles + reference_handles:
            h.remove()

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert custom_shapes_during_forward.get(parameter_name) == global_shape
            assert reference_shapes_during_forward.get(parameter_name) == global_shape

    def test_forward_parameters_resharded_after_forward_parity(self) -> None:
        """
        3. Verifies that parameters are resharded back to local shapes after forward in both
        models.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device)
        wrapped_custom(dummy_input)
        wrapped_reference(dummy_input)

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            reference_param = getattr(getattr(wrapped_reference, layer_name), attr)

            assert custom_param.to_local().shape == expected_local_shape
            assert reference_param.to_local().shape == expected_local_shape
            assert not wrapped_custom.sharded_parameters[parameter_name].is_gathered

    def test_backward_parameters_are_gathered_parity(self) -> None:
        """
        4. Verifies that parameters are all-gathered during the backward pass in both
        models.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        custom_shapes_during_backward: dict[str, torch.Size] = {}
        reference_shapes_during_backward: dict[str, torch.Size] = {}

        def make_hook(model: nn.Module, shapes_dict: dict[str, Size]):
            def hook(module: nn.Module, _grad_output: Tensor | tuple[Tensor, ...]):
                for parameter_name in self.expected_shapes:
                    layer_name, attr = parameter_name.split(".")
                    if module == getattr(model, layer_name):
                        param = getattr(module, attr)
                        shapes_dict[parameter_name] = param.shape

            return hook

        custom_handles = [
            getattr(
                wrapped_custom.inner_module, p.split(".")[0]
            ).register_full_backward_pre_hook(
                make_hook(
                    model=wrapped_custom.inner_module,
                    shapes_dict=custom_shapes_during_backward,
                )
            )
            for p in self.expected_shapes
        ]
        reference_handles = [
            getattr(wrapped_reference, p.split(".")[0]).register_full_backward_pre_hook(
                make_hook(
                    model=wrapped_reference,
                    shapes_dict=reference_shapes_during_backward,
                )
            )
            for p in self.expected_shapes
        ]

        dummy_input = torch.randn(
            2, self.base_model.input_dim, device=self.device, requires_grad=True
        )
        wrapped_custom(dummy_input).sum().backward()

        dummy_input_reference = dummy_input.clone().detach().requires_grad_(True)
        wrapped_reference(dummy_input_reference).sum().backward()

        for h in custom_handles + reference_handles:
            h.remove()

        for parameter_name, (global_shape, _) in self.expected_shapes.items():
            assert custom_shapes_during_backward.get(parameter_name) == global_shape
            assert reference_shapes_during_backward.get(parameter_name) == global_shape

    def test_backward_gradients_are_reduce_scattered_parity(self) -> None:
        """
        5. Verifies that gradients are reduce-scattered to local sharded shapes matching
        Native FSDP2.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        dummy_input = torch.randn(
            2, self.base_model.input_dim, device=self.device, requires_grad=True
        )
        wrapped_custom(dummy_input).sum().backward()

        dummy_input_reference = dummy_input.clone().detach().requires_grad_(True)
        wrapped_reference(dummy_input_reference).sum().backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            reference_param = getattr(getattr(wrapped_reference, layer_name), attr)

            assert custom_param.grad is not None
            assert reference_param.grad is not None

            custom_grad_local = custom_param.grad.to_local()
            reference_grad_local = reference_param.grad.to_local()

            assert (
                custom_grad_local.shape == reference_grad_local.shape == expected_local_shape
            )
            torch.testing.assert_close(
                custom_grad_local, reference_grad_local, rtol=1e-5, atol=1e-8
            )

    def test_parameters_resharded_after_backward_parity(self) -> None:
        """
        6. Verifies that parameters are resharded back to local shapes after backward in
        both models.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        wrapped_reference = _reference_fully_shard_model(
            copy.deepcopy(self.base_model), self.device_mesh
        )

        dummy_input = torch.randn(
            2, self.base_model.input_dim, device=self.device, requires_grad=True
        )
        wrapped_custom(dummy_input).sum().backward()

        dummy_input_reference = dummy_input.clone().detach().requires_grad_(True)
        wrapped_reference(dummy_input_reference).sum().backward()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            custom_param = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            reference_param = getattr(getattr(wrapped_reference, layer_name), attr)

            assert custom_param.to_local().shape == expected_local_shape
            assert reference_param.to_local().shape == expected_local_shape
            assert not wrapped_custom.sharded_parameters[parameter_name].is_gathered


@pytest.mark.multi_rank
class TestConvergenceParity(DistributedTestBase):
    """Verifies end-to-end training loop convergence parity against FSDP2."""

    def test_multi_step_training_convergence_parity(self) -> None:
        """
        Verifies forward outputs and loss values match Native FSDP2 over a 5-step SGD
        training loop.
        """
        wrapped_custom = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        custom_optimizer = torch.optim.SGD(wrapped_custom.parameters(), lr=0.01)

        reference_raw = copy.deepcopy(self.base_model)
        fully_shard(reference_raw.fc1, mesh=self.device_mesh)
        fully_shard(reference_raw.fc2, mesh=self.device_mesh)
        wrapped_reference = cast(
            "nn.Module", fully_shard(reference_raw, mesh=self.device_mesh)
        )
        reference_optimizer = torch.optim.SGD(wrapped_reference.parameters(), lr=0.01)

        torch.manual_seed(100 + self.rank)
        batch_size = 2
        x = torch.randn(
            batch_size, self.base_model.input_dim, device=self.device, requires_grad=True
        )
        target = torch.randn(batch_size, self.base_model.output_dim, device=self.device)
        criterion = nn.MSELoss()

        for step in range(1, 6):
            custom_optimizer.zero_grad()
            edu_out = wrapped_custom(x)
            edu_loss = criterion(edu_out, target)
            edu_loss.backward()
            custom_optimizer.step()

            reference_optimizer.zero_grad()
            reference_out = wrapped_reference(x)
            reference_loss = criterion(reference_out, target)
            reference_loss.backward()
            reference_optimizer.step()

            output_diff = torch.abs(edu_out - reference_out).max().item()
            loss_diff = abs(edu_loss.item() - reference_loss.item())

            assert output_diff < 1e-5, f"Forward output diverged at step {step}!"
            assert loss_diff < 1e-5, f"Loss value diverged at step {step}!"

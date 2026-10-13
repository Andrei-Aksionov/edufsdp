import copy

import pytest
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from gloofsdp.fsdp import FSDP
from gloofsdp.state_dict import StateDictType as EDUStateDictType
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestStateDictParity(DistributedTestBase):
    def test_state_dict(self):
        # 1. Base Model Initialization & Weight Sync
        # Wrap with EducationalFSDP and Native FSDP
        edu_model = FSDP(copy.deepcopy(self.base_model), device_mesh=self.device_mesh)
        native_model = fully_shard(copy.deepcopy(self.base_model), mesh=self.device_mesh)

        print(f"[Rank {self.rank}] Testing State Dict Extraction...")

        # ------------------------------------------------------------------
        # TEST A: FULL State Dict Parity (EducationalFSDP vs Native FSDP)
        # ------------------------------------------------------------------
        # Get Full State Dict from Educational FSDP (All-Gathers parameters into global shapes)
        edu_full_state_dict = edu_model.state_dict(state_dict_type=EDUStateDictType.FULL)

        # Get Full State Dict from Native PyTorch FSDP
        native_full_state_dict, _ = get_state_dict(
            native_model, optimizers=(), options=StateDictOptions(full_state_dict=True)
        )

        if self.rank == 0:
            assert set(edu_full_state_dict.keys()) == set(native_full_state_dict.keys()), (
                "State dict keys mismatch between EducationalFSDP and Native FSDP!"
            )

            for key in edu_full_state_dict:
                edu_tensor = edu_full_state_dict[key]
                native_tensor = native_full_state_dict[key]

                # Verify shapes match full un-sharded parameter shapes
                assert edu_tensor.shape == native_tensor.shape, (
                    f"Shape mismatch for {key}: Edu {edu_tensor.shape} vs Native {native_tensor.shape}"
                )

                # Verify values are identical
                max_diff = torch.abs(edu_tensor - native_tensor).max().item()
                assert max_diff < 1e-6, f"Value mismatch in full state_dict for {key}: max diff {max_diff}"


@pytest.mark.multi_rank(world_size=2)
class TestDTensorConsistency(DistributedTestBase):
    def test_dtensor_parity_against_native_fsdp2(self) -> None:
        """Verifies complete DTensor property parity (type, placement, mesh, grad type) against Native FSDP2."""
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = FSDP(model_custom, device_mesh=self.device_mesh)

        model_native = copy.deepcopy(self.base_model)
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


@pytest.mark.multi_rank(world_size=2)
class TestOptimizerStates(DistributedTestBase):
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    def test_optimizer_state_parity_post_step(self) -> None:
        """Verifies optimizer state lazy creation, local sharded shapes, and parity against FSDP2 after step()."""
        # 1. Setup Custom FSDP
        model_custom = copy.deepcopy(self.base_model)
        wrapped_custom = FSDP(model_custom, device_mesh=self.device_mesh)
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
        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")

            custom_parameter = getattr(getattr(wrapped_custom.inner_module, layer_name), attr)
            native_parameter = getattr(getattr(model_native, layer_name), attr)

            # Ensure parameters are tracked inside optimizer state dicts
            assert custom_parameter in optimizer_custom.state, (
                f"Rank {self.rank}: {parameter_name} was not populated in custom optimizer.state"
            )
            assert native_parameter in optimizer_native.state, (
                f"Rank {self.rank}: {parameter_name} was not populated in native optimizer.state"
            )

            custom_state = optimizer_custom.state[custom_parameter]
            native_state = optimizer_native.state[native_parameter]

            # Verify Adam keys exist
            for state_key in ("exp_avg", "exp_avg_sq"):
                assert state_key in custom_state, (
                    f"Rank {self.rank}: Key '{state_key}' missing from custom optimizer state for {parameter_name}"
                )

                custom_tensor = (
                    custom_state[state_key].to_local()
                    if hasattr(custom_state[state_key], "to_local")
                    else custom_state[state_key]
                )
                native_tensor = (
                    native_state[state_key].to_local()
                    if hasattr(native_state[state_key], "to_local")
                    else native_state[state_key]
                )

                # Assert shape matches expected local sharded shape and native FSDP shape
                assert custom_tensor.shape == native_tensor.shape == expected_local_shape, (
                    f"Rank {self.rank}: {parameter_name} state '{state_key}' shape mismatch - Custom "
                    f"{custom_tensor.shape}, Native {native_tensor.shape}, Expected {expected_local_shape}"
                )


@pytest.mark.multi_rank(world_size=2)
class TestShardingPlacement(DistributedTestBase):
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

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


@pytest.mark.multi_rank(world_size=2)
class TestLossParity(DistributedTestBase):
    def test_loss_parity(self):
        # Make deepcopies so both FSDP implementations start from identical parameters
        edu_raw_model = copy.deepcopy(self.base_model)
        native_raw_model = copy.deepcopy(self.base_model)

        # 2. Wrap Models
        # Educational FSDP (gloofsdp)
        edu_model = FSDP(edu_raw_model, device_mesh=self.device_mesh)
        edu_optimizer = torch.optim.SGD(edu_model.parameters(), lr=0.01)

        # PyTorch Native FSDP
        native_model = fully_shard(native_raw_model, mesh=self.device_mesh)
        native_optimizer = torch.optim.SGD(native_model.parameters(), lr=0.01)

        # 3. Create Deterministic Per-Rank Input Data & Target Data
        # Fixed seed for inputs so run-to-run comparisons are identical
        torch.manual_seed(100 + self.rank)
        # requires_grad=True is required for the input tensor because module-level backward
        # hooks (register_full_backward_pre_hook) on the first layer need an entry autograd node.
        # Without it, autograd prunes the input graph edge and emits a UserWarning when firing
        # backward hooks at the output node instead.
        batch_size = 2
        x = torch.randn(batch_size, self.base_model.input_dim, device=self.device, requires_grad=True)
        target = torch.randn(batch_size, self.base_model.output_dim, device=self.device)
        criterion = nn.MSELoss()

        print(f"[Rank {self.rank}] Running 5-step convergence comparison...")

        # 4. Training Step Verification Loop
        for step in range(1, 6):
            # --- Educational FSDP Step ---
            edu_optimizer.zero_grad()
            edu_out = edu_model(x)
            edu_loss = criterion(edu_out, target)
            edu_loss.backward()
            edu_optimizer.step()

            # --- Native PyTorch FSDP Step ---
            native_optimizer.zero_grad()
            native_out = native_model(x)
            native_loss = criterion(native_out, target)
            native_loss.backward()
            native_optimizer.step()

            # --- Assertions & Checks ---
            # A. Check Forward Pass Output Matching
            output_diff = torch.abs(edu_out - native_out).max().item()

            # B. Check Loss Value Matching
            loss_diff = abs(edu_loss.item() - native_loss.item())

            print(f"[Rank {self.rank}] Step {step} | Forward Max Diff: {output_diff:.2e} | Loss Diff: {loss_diff:.2e}")

            # Tolerances for FP32 eager operations
            assert output_diff < 1e-5, f"Forward output diverged at step {step}!"
            assert loss_diff < 1e-5, f"Loss value diverged at step {step}!"

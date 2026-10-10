"""
Unit tests verifying parameter placement, lifecycle shape transitions,
and FSDP2 DTensor parity across all layers.
"""

import copy

import pytest
import torch
from torch.distributed.fsdp import fully_shard

from gloofsdp.fsdp import FSDP
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestOptimizerStates(DistributedTestBase):
    """Test suite covering static sharding, runtime weight lifecycle, and FSDP2 parity."""

    def test_optimizer_state_empty_before_step(self) -> None:
        """Verifies that optimizer states remain unitialized/empty across init, forward, and backward."""
        model = copy.deepcopy(self.base_model)
        wrapped = FSDP(model, device_mesh=self.device_mesh)
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

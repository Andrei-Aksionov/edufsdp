"""Integration tests verifying that optimizer states are lazily created as sharded tensors,
confirming that states applied to already sharded FSDP parameters naturally inherit local sharded
shapes without requiring manual reallocation or re-sharding.
"""

import copy

import pytest
import torch

from gloofsdp.fsdp import FSDP
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestOptimizerStates(DistributedTestBase):
    """Test suite covering optimizer state initialization, lazy population, and sharded shape consistency."""

    def test_optimizer_state_empty_before_step(self) -> None:
        """Verifies that optimizer states remain uninitialized across init, forward, and backward passes."""
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)

        # 1. State should be empty right after instantiation
        assert len(optimizer.state) == 0, "Optimizer state should be empty after initialization"

        # 2. State should remain empty post-forward
        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device, requires_grad=True)
        output = wrapped_model(dummy_input)
        assert len(optimizer.state) == 0, "Optimizer state should be empty post-forward"

        # 3. State should remain empty post-backward (before step)
        loss = output.sum()
        loss.backward()
        assert len(optimizer.state) == 0, "Optimizer state should be empty post-backward (before step)"

    def test_optimizer_state_populated_post_step(self) -> None:
        """Verifies Adam optimizer state lazy creation, state keys, and local sharded shapes after step()."""
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device, requires_grad=True)
        output = wrapped_model(dummy_input)
        output.sum().backward()
        optimizer.step()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            parameter = getattr(getattr(wrapped_model.inner_module, layer_name), attr)

            # Ensure parameter is tracked inside optimizer state dict
            assert parameter in optimizer.state, (
                f"Rank {self.rank}: {parameter_name} was not populated in optimizer.state"
            )

            state = optimizer.state[parameter]

            # Verify Adam state keys exist and match expected local sharded shapes
            for state_key in ("exp_avg", "exp_avg_sq"):
                assert state_key in state, (
                    f"Rank {self.rank}: Key '{state_key}' missing from optimizer state for {parameter_name}"
                )

                tensor_val = state[state_key].to_local() if hasattr(state[state_key], "to_local") else state[state_key]

                assert tensor_val.shape == expected_local_shape, (
                    f"Rank {self.rank}: {parameter_name} state '{state_key}' shape mismatch - "
                    f"Got {tensor_val.shape}, Expected {expected_local_shape}"
                )

    def test_sgd_momentum_optimizer_state_populated_post_step(self) -> None:
        """Verifies non-Adam optimizer (SGD with momentum) state lazy creation and local sharded shapes after step()."""
        model = copy.deepcopy(self.base_model)
        wrapped_model = FSDP(model, device_mesh=self.device_mesh)
        optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=0.01, momentum=0.9)

        dummy_input = torch.randn(2, self.base_model.input_dim, device=self.device, requires_grad=True)
        output = wrapped_model(dummy_input)
        output.sum().backward()
        optimizer.step()

        for parameter_name, (_, expected_local_shape) in self.expected_shapes.items():
            layer_name, attr = parameter_name.split(".")
            parameter = getattr(getattr(wrapped_model.inner_module, layer_name), attr)

            # Ensure parameter is tracked inside optimizer state dict
            assert parameter in optimizer.state, (
                f"Rank {self.rank}: {parameter_name} was not populated in SGD optimizer.state"
            )

            state = optimizer.state[parameter]

            # Verify SGD momentum buffer key exists and matches expected local sharded shape
            assert "momentum_buffer" in state, (
                f"Rank {self.rank}: Key 'momentum_buffer' missing from SGD optimizer state for {parameter_name}"
            )

            tensor_val = (
                state["momentum_buffer"].to_local()
                if hasattr(state["momentum_buffer"], "to_local")
                else state["momentum_buffer"]
            )

            assert tensor_val.shape == expected_local_shape, (
                f"Rank {self.rank}: {parameter_name} state 'momentum_buffer' shape mismatch - "
                f"Got {tensor_val.shape}, Expected {expected_local_shape}"
            )

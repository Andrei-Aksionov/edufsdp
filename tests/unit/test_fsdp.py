"""Unit tests for gloofsdp.fsdp covering module wrapping, parameter traversal, hooks, and state dict delegation."""

from unittest.mock import patch

import pytest
import torch

from gloofsdp.fsdp import FSDP
from gloofsdp.hooks import BackwardPostHook, BackwardPreHook, ForwardPostHook, ForwardPreHook
from gloofsdp.sharded_parameter import ShardedParameter
from gloofsdp.state_dict import StateDictType
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank(world_size=2)
class TestFSDP(DistributedTestBase):
    """Test suite for FSDP wrapper tree wrapping, parameter collection, hook registration, and delegation."""

    def test_sharded_parameters_property_completeness(self) -> None:
        """Verifies that the sharded_parameters property correctly traverses the module tree and maps all parameters."""
        fsdp_model = FSDP(self.base_model, self.device_mesh)
        sharded_map = fsdp_model.sharded_parameters

        expected_fqns = {"fc1.weight", "fc2.weight"}
        assert set(sharded_map.keys()) == expected_fqns

        for fqn, sharded_parameter in sharded_map.items():
            assert isinstance(sharded_parameter, ShardedParameter)
            assert sharded_parameter.fqn == fqn
            _, expected_local_shape = self.expected_shapes[fqn]
            assert sharded_parameter.local_shard.shape == expected_local_shape

    def test_hook_registration_on_submodules(self) -> None:
        """Verifies that exact FSDP hook instances are registered on submodules owning parameters."""
        fsdp_model = FSDP(self.base_model, self.device_mesh)

        for name, submodule in fsdp_model.inner_module.named_modules():
            # Skip container modules that don't own direct parameters
            if not len(submodule._fsdp_sharded_parameters):
                continue

            # Verify specific FSDP hook instance types exist among registered hooks
            pre_hooks = list(submodule._forward_pre_hooks.values())
            post_hooks = list(submodule._forward_hooks.values())
            backward_pre_hooks = list(submodule._backward_pre_hooks.values())
            backward_post_hooks = list(submodule._backward_hooks.values())

            assert any(isinstance(hook, ForwardPreHook) for hook in pre_hooks)
            assert any(isinstance(hook, ForwardPostHook) for hook in post_hooks)
            assert any(isinstance(hook, BackwardPreHook) for hook in backward_pre_hooks)
            assert any(isinstance(hook, BackwardPostHook) for hook in backward_post_hooks)

    @patch("gloofsdp.fsdp.StateDictManager.get_state_dict")
    def test_state_dict_delegation_passes_all_sharded_parameters(self, mock_get_state_dict) -> None:
        """Verifies that fsdp.state_dict delegates to StateDictManager without skipping any sharded parameters."""
        fsdp_model = FSDP(self.base_model, self.device_mesh)
        mock_get_state_dict.return_value = {"mock": "state_dict"}

        result = fsdp_model.state_dict(StateDictType.LOCAL)

        mock_get_state_dict.assert_called_once()
        _, passed_sharded_parameters, passed_type = mock_get_state_dict.call_args[0]

        # Verify no sharded parameters were skipped or omitted
        assert set(passed_sharded_parameters.keys()) == {"fc1.weight", "fc2.weight"}
        assert passed_type == StateDictType.LOCAL
        assert result == {"mock": "state_dict"}

    @patch("gloofsdp.fsdp.StateDictManager.load_state_dict")
    def test_load_state_dict_delegation_passes_all_sharded_parameters(self, mock_load_state_dict) -> None:
        """Verifies that fsdp.load_state_dict delegates to StateDictManager with complete sharded parameters."""
        fsdp_model = FSDP(self.base_model, self.device_mesh)
        dummy_state_dict = {"fc1.weight": torch.zeros(2, 16)}

        fsdp_model.load_state_dict(dummy_state_dict, StateDictType.LOCAL)

        mock_load_state_dict.assert_called_once()
        _, passed_sharded_parameters, passed_dict, passed_type = mock_load_state_dict.call_args[0]

        assert set(passed_sharded_parameters.keys()) == {"fc1.weight", "fc2.weight"}
        assert passed_dict == dummy_state_dict
        assert passed_type == StateDictType.LOCAL

    def test_fsdp_forward_and_backward_execution(self) -> None:
        """Verifies end-to-end forward and backward passes execute correctly through the FSDP wrapper."""
        fsdp_model = FSDP(self.base_model, self.device_mesh)
        batch_size = 4

        input_tensor = torch.randn(batch_size, 16)

        # 1. Forward Pass
        output = fsdp_model(input_tensor)
        assert output.shape == torch.Size([batch_size, 8])

        # 2. Backward Pass
        loss = output.sum()
        loss.backward()

        # 3. Verify gradients are correctly reduced and sharded across ranks
        for fqn, sharded_parameter in fsdp_model.sharded_parameters.items():
            parameter = getattr(fsdp_model.inner_module, fqn.split(".")[0]).weight
            assert parameter.grad is not None
            _, expected_local_shape = self.expected_shapes[fqn]
            assert parameter.grad.to_local().shape == expected_local_shape

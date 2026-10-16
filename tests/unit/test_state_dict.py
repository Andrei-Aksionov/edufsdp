"""
Unit tests for gloofsdp.state_dict covering LOCAL and FULL state dict extraction,
loading, and parameter state preservation.
"""

import copy

import pytest
import torch
from torch.distributed.tensor import DTensor

from gloofsdp.fsdp import FSDP
from gloofsdp.state_dict import StateDictManager, StateDictType
from tests.base.test_class import DistributedTestBase


@pytest.mark.multi_rank
class TestStateDictManager(DistributedTestBase):
    """
    Test suite for StateDictManager handling local shards, full checkpoints,
    and parameter isolation.
    """

    def test_get_state_dict_local_format(self) -> None:
        """
        Verifies extraction of LOCAL state dict returns rank-local shard shapes
        and values.
        """
        fsdp_model = FSDP(self.base_model, self.device_mesh)
        state_dict = StateDictManager.get_state_dict(
            fsdp_model.inner_module,
            fsdp_model.sharded_parameters,
            StateDictType.LOCAL,
        )

        assert set(state_dict.keys()) == {"fc1.weight", "fc2.weight"}

        for fqn, tensor in state_dict.items():
            _, expected_local_shape = self.expected_shapes[fqn]
            assert isinstance(tensor, DTensor)
            assert tensor.to_local().shape == expected_local_shape

            # Verify values match local shard storage directly
            expected_local_shard = fsdp_model.sharded_parameters[fqn].local_shard
            torch.testing.assert_close(tensor.to_local(), expected_local_shard)

    def test_get_state_dict_full_format_matches_baseline(self) -> None:
        """
        Verifies extraction of FULL state dict consolidates global tensors
        matching the unsharded baseline.
        """
        fsdp_model = FSDP(copy.deepcopy(self.base_model), self.device_mesh)
        full_state_dict = StateDictManager.get_state_dict(
            fsdp_model.inner_module,
            fsdp_model.sharded_parameters,
            StateDictType.FULL,
        )

        assert set(full_state_dict.keys()) == {"fc1.weight", "fc2.weight"}

        # Compare against original reference base_model parameters
        for fqn, tensor in full_state_dict.items():
            global_shape, _ = self.expected_shapes[fqn]
            assert tensor.shape == global_shape

            baseline_parameter = getattr(self.base_model, fqn.split(".")[0]).weight
            torch.testing.assert_close(tensor, baseline_parameter)

    def test_get_state_dict_full_preserves_sharded_parameter_state(self) -> None:
        """
        Verifies that extracting a FULL state dict gathers tensors during extraction
        but leaves parameters sharded.
        """
        fsdp_model = FSDP(self.base_model, self.device_mesh)

        # Confirm all parameters start sharded
        for sharded_parameter in fsdp_model.sharded_parameters.values():
            assert not sharded_parameter.is_gathered

        # Extract FULL state dict
        full_state_dict = StateDictManager.get_state_dict(
            fsdp_model.inner_module,
            fsdp_model.sharded_parameters,
            StateDictType.FULL,
        )

        # Assert returned state dict contains full tensors
        for fqn, tensor in full_state_dict.items():
            global_shape, _ = self.expected_shapes[fqn]
            assert tensor.shape == global_shape

        # Verify that underlying ShardedParameter instances remain
        # in sharded state (is_gathered=False)
        for sharded_parameter in fsdp_model.sharded_parameters.values():
            assert not sharded_parameter.is_gathered
            _, expected_local_shape = self.expected_shapes[sharded_parameter.fqn]
            assert sharded_parameter.shape == expected_local_shape

    def test_load_state_dict_local_format(self) -> None:
        """Verifies reloading LOCAL format state dict updates rank-local shards correctly."""
        fsdp_model_src = FSDP(copy.deepcopy(self.base_model), self.device_mesh)
        local_state_dict = StateDictManager.get_state_dict(
            fsdp_model_src.inner_module,
            fsdp_model_src.sharded_parameters,
            StateDictType.LOCAL,
        )

        # Modify source weights slightly
        with torch.no_grad():
            for tensor in local_state_dict.values():
                tensor.add_(5.0)

        fsdp_model_target = FSDP(copy.deepcopy(self.base_model), self.device_mesh)
        StateDictManager.load_state_dict(
            fsdp_model_target.inner_module,
            fsdp_model_target.sharded_parameters,
            local_state_dict,
            StateDictType.LOCAL,
        )

        # Verify target shards reflect updated values
        target_local_state_dict = StateDictManager.get_state_dict(
            fsdp_model_target.inner_module,
            fsdp_model_target.sharded_parameters,
            StateDictType.LOCAL,
        )
        for fqn in local_state_dict:
            torch.testing.assert_close(target_local_state_dict[fqn], local_state_dict[fqn])

    def test_load_state_dict_full_format(self) -> None:
        """
        Verifies reloading FULL format state dict
        correctly redistributes global tensors across ranks.
        """
        fsdp_model_src = FSDP(copy.deepcopy(self.base_model), self.device_mesh)
        full_state_dict = StateDictManager.get_state_dict(
            fsdp_model_src.inner_module,
            fsdp_model_src.sharded_parameters,
            StateDictType.FULL,
        )

        fsdp_model_target = FSDP(copy.deepcopy(self.base_model), self.device_mesh)
        StateDictManager.load_state_dict(
            fsdp_model_target.inner_module,
            fsdp_model_target.sharded_parameters,
            full_state_dict,
            StateDictType.FULL,
        )

        # Verify target extracted full state dict matches source full state dict
        target_full_state_dict = StateDictManager.get_state_dict(
            fsdp_model_target.inner_module,
            fsdp_model_target.sharded_parameters,
            StateDictType.FULL,
        )
        for fqn in full_state_dict:
            torch.testing.assert_close(target_full_state_dict[fqn], full_state_dict[fqn])

    def test_load_state_dict_raises_key_error_on_unknown_parameter(self) -> None:
        """
        Verifies KeyError is raised when incoming state dict
        contains a parameter not present in module.
        """
        fsdp_model = FSDP(self.base_model, self.device_mesh)
        invalid_state_dict = {"nonexistent.weight": torch.zeros(2, 2)}

        with pytest.raises(KeyError):
            StateDictManager.load_state_dict(
                fsdp_model.inner_module,
                fsdp_model.sharded_parameters,
                invalid_state_dict,
                StateDictType.LOCAL,
            )

"""
State dictionary management for Educational FSDP.

Handles extracting and loading model state dictionaries in two formats:
1. `LOCAL`: Saves/loads rank-local 1/N parameter tensor shards.
2. `FULL`: Saves/loads consolidated global parameters (unsharded).
"""

from enum import Enum

import torch
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor

from gloofsdp.hooks import FSDPAllGatherFunction
from gloofsdp.sharded_parameter import ShardedParameter


class StateDictType(Enum):
    """Supported state dict format types."""

    FULL = "full"  # Consolidated global parameters across ranks
    LOCAL = "local"  # Rank-local sharded parameter tensors


class StateDictManager:
    """Manages state dict extraction and restoration for sharded models."""

    @staticmethod
    def _get_sharded_parameter_map(module: nn.Module) -> dict[str, ShardedParameter]:
        """
        Dynamically collects all DTensorShardedParameter objects across
        submodules by searching through module attributes.
        """
        sharded_map: dict[str, ShardedParameter] = {}

        for module_name, submodule in module.named_modules():
            # Check if this submodule holds any registered sharded parameter objects
            if hasattr(submodule, "_fsdp_sharded_parameters"):
                for parameter_name, parameter_wrapper in submodule._fsdp_sharded_parameters.items():
                    # Construct fully qualified parameter name (e.g., 'layer1.weight')
                    full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                    sharded_map[full_name] = parameter_wrapper

        return sharded_map

    @staticmethod
    @torch.no_grad()
    def get_state_dict(
        module: nn.Module,
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> dict[str, torch.Tensor]:
        """Extracts module state dictionary in either `LOCAL` or `FULL` format.

        Args:
            module: Target `nn.Module` wrapped by FSDP.
            state_dict_type: Target format (`StateDictType.LOCAL` or `StateDictType.FULL`).

        Returns:
            dict[str, torch.Tensor]: Map of parameter FQNs to local shards or full tensors.
        """
        state_dict: dict[str, torch.Tensor] = {}
        sharded_parameters = StateDictManager._get_sharded_parameter_map(module)

        for parameter_name, parameter in module.named_parameters():
            if parameter_name not in sharded_parameters:
                tensor = parameter.data
            else:
                parameter = sharded_parameters[parameter_name]

                if state_dict_type == StateDictType.LOCAL:
                    tensor = parameter.local_shard
                else:
                    tensor = (
                        parameter._gathered_tensor
                        if parameter.is_gathered
                        else FSDPAllGatherFunction.apply(parameter.local_shard, parameter)
                    )

            # Store detached copy to prevent state contamination
            state_dict[parameter_name] = tensor.detach().clone()

        return state_dict

    @staticmethod
    @torch.no_grad()
    def load_state_dict(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> None:
        """Loads a state dictionary checkpoint into sharded model parameters.

        Args:
            module: Target `nn.Module` wrapped by FSDP.
            state_dict: Incoming map of parameter FQNs to tensors.
            state_dict_type: Format of incoming state_dict (`LOCAL` or `FULL`).

        Raises:
            KeyError: If a parameter in `state_dict` does not exist in `module`.
        """

        module_named_parameters = dict(module.named_parameters())
        sharded_parameters = StateDictManager._get_sharded_parameter_map(module)

        for sd_name, sd_tensor in state_dict.items():
            if sd_name not in module_named_parameters:
                raise KeyError(f"Parameter '{sd_name}' from state_dict cannot be found in the module.")

            # Handle non-sharded parameters
            if sd_name not in sharded_parameters:
                target_parameter = module_named_parameters[sd_name]
                target_parameter.copy_(sd_tensor.to(target_parameter.device))
                continue

            parameter = sharded_parameters[sd_name]

            if state_dict_type == StateDictType.LOCAL:
                # Copy incoming local shard directly into local buffer
                parameter.local_shard.copy_(sd_tensor.to(parameter.local_shard.device))

            elif state_dict_type == StateDictType.FULL:
                # Distribute global tensor across mesh to extract current rank's shard
                sharded_dtensor = distribute_tensor(
                    sd_tensor.to(parameter.local_shard.device),
                    device_mesh=parameter.device_mesh,
                    placements=[Shard(dim=0)],
                )

                # Copy shard data into the existing rank-local parameter buffer
                parameter.local_shard.copy_(sharded_dtensor.to_local())

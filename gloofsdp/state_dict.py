"""
State dictionary management for Educational FSDP.

Handles extracting and loading model state dictionaries in two formats:
1. `LOCAL`: Saves/loads rank-local 1/N parameter tensor shards.
2. `FULL`: Saves/loads consolidated global parameters (unsharded).
"""

from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import Shard, distribute_tensor

from gloofsdp.hooks import FSDPAllGatherFunction

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gloofsdp.sharded_parameter import ShardedParameter


class StateDictType(Enum):
    """Supported state dict format types."""

    FULL = "full"  # Consolidated global parameters across ranks
    LOCAL = "local"  # Rank-local sharded parameter tensors


class StateDictManager:
    """Manages state dict extraction and restoration for sharded models."""

    @staticmethod
    @torch.no_grad()
    def get_state_dict(
        module: nn.Module,
        sharded_parameters: dict[str, ShardedParameter],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> dict[str, Tensor]:
        """
        Extracts module state dictionary in either `LOCAL` or `FULL` format.

        Args:
            module: Target `nn.Module` wrapped by FSDP.
            sharded_parameters: Dictionary with sharded parameters of the target module.
            state_dict_type: Target format (`StateDictType.LOCAL` or `StateDictType.FULL`).

        Returns:
            Map of parameter FQNs to local shards or full tensors.

        """
        state_dict: dict[str, Tensor] = {}

        for parameter_name, parameter in module.named_parameters():
            if parameter_name not in sharded_parameters:
                tensor = parameter.data
            else:
                parameter = sharded_parameters[parameter_name]

                if state_dict_type == StateDictType.LOCAL:
                    tensor = parameter.local_shard
                else:
                    gathered = parameter._gathered_tensor
                    tensor = (
                        gathered
                        if gathered is not None
                        else FSDPAllGatherFunction.apply(parameter.local_shard, parameter)
                    )

            # Store detached copy to prevent state contamination
            state_dict[parameter_name] = tensor.detach().clone()

        return state_dict

    @staticmethod
    @torch.no_grad()
    def load_state_dict(
        module: nn.Module,
        sharded_parameters: dict[str, ShardedParameter],
        state_dict: Mapping[str, Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> None:
        """
        Loads a state dictionary checkpoint into sharded model parameters.

        Args:
            module: Target `nn.Module` wrapped by FSDP.
            sharded_parameters: Dictionary with sharded parameters of the target module.
            state_dict: Incoming map of parameter FQNs to tensors.
            state_dict_type: Format of incoming state_dict (`LOCAL` or `FULL`).

        Raises:
            KeyError: If a parameter in `state_dict` does not exist in `module`.

        """
        module_named_parameters = dict(module.named_parameters())

        for sd_name, sd_tensor in state_dict.items():
            if sd_name not in module_named_parameters:
                raise KeyError(
                    f"Parameter '{sd_name}' from state_dict cannot be found in the module."
                )

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

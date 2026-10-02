from enum import Enum
from typing import Dict

import torch
from torch import nn
from torch.distributed.tensor import Shard, distribute_tensor

from gloofsdp.hooks import FSDPAllGatherFunction
from gloofsdp.sharded_parameter import DTensorShardedParameter


class StateDictType(Enum):
    """Supported state dict types."""

    FULL = "full"  # Consolidated global parameters across ranks
    LOCAL = "local"  # Rank-local sharded parameter tensors


class StateDictManager:
    """Manages state dict extraction and restoration for sharded models."""

    @staticmethod
    @torch.no_grad()
    def get_state_dict(
        module: nn.Module,
        sharded_params: dict[str, DTensorShardedParameter],
        state_dict_type: StateDictType,
    ) -> dict[str, torch.Tensor]:
        """
        Extracts state dict according to the active `state_dict_type`.

        Args:
            module: Target `nn.Module`.
            sharded_params: Mapping of parameter names to DTensorShardedParameter wrappers.

        Returns:
            dict[str, torch.Tensor]: State dict mapping names to full or local tensors.
        """
        state_dict: dict[str, torch.Tensor] = {}

        for name, param in module.named_parameters():
            if name not in sharded_params:
                tensor = param.data
            else:
                p = sharded_params[name]

                if state_dict_type == StateDictType.LOCAL:
                    tensor = p.local_shard
                else:  # StateDictType.FULL
                    tensor = p._full_tensor if p.is_gathered else FSDPAllGatherFunction.apply(p)

            state_dict[name] = tensor.detach().clone()

        return state_dict

    @staticmethod
    @torch.no_grad()
    def load_state_dict(
        module: nn.Module,
        sharded_params: Dict[str, DTensorShardedParameter],
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType,
    ) -> None:

        module_keys = dict(module.named_parameters())

        for name, incoming_tensor in state_dict.items():
            if name not in module_keys:
                raise KeyError(f"Parameter '{name}' from state_dict cannot be found in the model.")

            if name not in sharded_params:
                target_param = module_keys[name]
                target_param.copy_(incoming_tensor.to(target_param.device))
                continue

            p = sharded_params[name]
            if state_dict_type == StateDictType.LOCAL:
                # Assign incoming local shard tensor directly
                p.local_shard.copy_(incoming_tensor.to(p.local_shard.device))

            elif state_dict_type == StateDictType.FULL:
                # Distribute the global tensor to get the correct rank-local shard
                sharded_dtensor = distribute_tensor(
                    incoming_tensor.to(p.local_shard.device),
                    device_mesh=p.device_mesh,
                    placements=[Shard(dim=0)],
                )

                # Copy data into the existing parameter buffer
                p.local_shard.copy_(sharded_dtensor.to_local())

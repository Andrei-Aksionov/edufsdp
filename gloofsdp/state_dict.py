from enum import Enum
from typing import Dict

import torch
from torch import nn
from torch.distributed.tensor import Shard, distribute_tensor

from gloofsdp.hooks import FSDPAllGatherFunction
from gloofsdp.sharded_parameter import ShardedParameter


class StateDictType(Enum):
    """Supported state dict types."""

    FULL = "full"  # Consolidated global parameters across ranks
    LOCAL = "local"  # Rank-local sharded parameter tensors


class StateDictManager:
    """Manages state dict extraction and restoration for sharded models."""

    @staticmethod
    def _get_sharded_param_map(module: nn.Module) -> Dict[str, ShardedParameter]:
        """
        Dynamically collects all DTensorShardedParameter objects across
        submodules by searching through module attributes.
        """
        sharded_map: Dict[str, ShardedParameter] = {}

        for mod_name, submodule in module.named_modules():
            # Check if this submodule holds any registered sharded parameter objects
            if hasattr(submodule, "_fsdp_sharded_params"):
                for p_name, param_wrapper in submodule._fsdp_sharded_params.items():
                    # Construct fully qualified parameter name (e.g., 'layer1.weight')
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    sharded_map[full_name] = param_wrapper

        return sharded_map

    @staticmethod
    @torch.no_grad()
    def get_state_dict(
        module: nn.Module,
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
        sharded_params = StateDictManager._get_sharded_param_map(module)

        for name, param in module.named_parameters():
            if name not in sharded_params:
                tensor = param.data
            else:
                p = sharded_params[name]

                if state_dict_type == StateDictType.LOCAL:
                    tensor = p.local_shard
                else:  # StateDictType.FULL
                    # TODO: I don't like this. This should retrieve data from the parameter attached to the module.
                    # So, ideally, the code should check that if it's not gathered yet, call the gather function first.
                    tensor = p._gathered_tensor if p.is_gathered else FSDPAllGatherFunction.apply(p.local_shard, p)

            state_dict[name] = tensor.detach().clone()

        return state_dict

    @staticmethod
    @torch.no_grad()
    def load_state_dict(
        module: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType,
    ) -> None:

        module_keys = dict(module.named_parameters())
        sharded_params = StateDictManager._get_sharded_param_map(module)

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

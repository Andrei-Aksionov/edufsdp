from enum import Enum
from typing import Dict

import torch
from torch.distributed.tensor import Shard, distribute_tensor

from gloofsdp.sharded_parameter import DTensorShardedParameter


class StateDictType(Enum):
    LOCAL = "local"
    FULL = "full"


class StateDictHandler:
    """
    Manages exporting and loading model state dictionaries in LOCAL or FULL formats.
    """

    @staticmethod
    def get_state_dict(
        sharded_params: Dict[str, DTensorShardedParameter],
        state_dict_type: StateDictType,
    ) -> Dict[str, torch.Tensor]:
        result = {}

        for name, param_wrapper in sharded_params.items():
            if state_dict_type == StateDictType.LOCAL:
                # Export rank-local shard tensor
                result[name] = param_wrapper.sharded_param.to_local().detach().cpu()

            elif state_dict_type == StateDictType.FULL:
                # 1. Ensure the full tensor is gathered across all ranks
                full_tensor = param_wrapper.gather()

                # 2. Detach and move to CPU so all ranks capture a valid unified tensor copy
                result[name] = full_tensor.detach().cpu()

                # 3. Release memory if it wasn't already active
                param_wrapper.free()

        return result

    @staticmethod
    def load_state_dict(
        sharded_params: Dict[str, DTensorShardedParameter],
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType,
    ) -> None:
        for name, param_wrapper in sharded_params.items():
            if name not in state_dict:
                raise KeyError(f"Parameter '{name}' not found in provided state dict.")

            incoming_tensor = state_dict[name]

            if state_dict_type == StateDictType.LOCAL:
                # Assign incoming local shard tensor directly
                with torch.no_grad():
                    local_shard = param_wrapper.sharded_param.to_local()
                    local_shard.copy_(incoming_tensor.to(local_shard.device))

            elif state_dict_type == StateDictType.FULL:
                # Incoming tensor is the global full-size tensor.
                # We need to slice/shard it appropriately for the current rank.
                device_mesh = param_wrapper.device_mesh

                with torch.no_grad():
                    # Distribute the global tensor to get the correct rank-local shard
                    sharded_dtensor = distribute_tensor(
                        incoming_tensor.to(param_wrapper.sharded_param.device),
                        device_mesh=device_mesh,
                        placements=[Shard(dim=0)],
                    )

                    # Copy data into the existing parameter buffer
                    local_shard = param_wrapper.sharded_param.to_local()
                    local_shard.copy_(sharded_dtensor.to_local())

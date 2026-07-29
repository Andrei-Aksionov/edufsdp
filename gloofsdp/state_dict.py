from enum import Enum, auto
from typing import Any, Dict

import torch
import torch.distributed as dist


class StateDictType(Enum):
    LOCAL = auto()
    FULL = auto()


class StateDictHandler:
    @staticmethod
    def get_state_dict(
        sharded_params: Dict[str, Any],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieves the state dict in either LOCAL or FULL format.
        """
        state_dict: Dict[str, torch.Tensor] = {}

        if state_dict_type == StateDictType.LOCAL:
            for name, param in sharded_params.items():
                # Save just the local parameter shard
                state_dict[name] = param.local_param.data.clone()

        elif state_dict_type == StateDictType.FULL:
            for name, param in sharded_params.items():
                # Gather the full parameter tensor across all ranks
                gathered_tensor = param.gather().data.clone()

                # Rank 0 collects and stores the full parameter
                if dist.get_rank() == 0:
                    state_dict[name] = gathered_tensor

                # Immediately free the gathered parameter to keep memory lean
                param.free()

        return state_dict

    @staticmethod
    def load_state_dict(
        sharded_params: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> None:
        """
        Loads a state dict back into the sharded parameters.
        """
        if state_dict_type == StateDictType.LOCAL:
            for name, param in sharded_params.items():
                if name in state_dict:
                    param.local_param.data.copy_(state_dict[name])

        elif state_dict_type == StateDictType.FULL:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            for name, param in sharded_params.items():
                full_tensor = state_dict[name].clone() if rank == 0 else torch.empty(param.global_shape)

                # Broadcast full tensor from rank 0 to all other ranks
                dist.broadcast(full_tensor, src=0)

                # Reshard the broadcasted tensor into the local rank's shard
                flat_param = full_tensor.flatten()
                if param.padding_needed > 0:
                    flat_param = torch.cat([flat_param, torch.zeros(param.padding_needed)])

                chunks = torch.chunk(flat_param, world_size)
                param.local_param.data.copy_(chunks[rank])

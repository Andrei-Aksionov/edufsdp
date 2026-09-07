from enum import Enum, auto
from typing import Any, Dict

import torch
import torch.distributed as dist

# Native DTensor imports (used if available)
# TODO (andrei aksionau): seems like it's not needed
try:
    from torch.distributed.tensor import Shard, distribute_tensor

    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False


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
                if hasattr(param, "dtensor"):
                    # DTensor version: extract local shard
                    state_dict[name] = param.dtensor.to_local().data.clone()
                else:
                    # Manual ShardedParameter version
                    state_dict[name] = param.local_param.data.clone()

        elif state_dict_type == StateDictType.FULL:
            for name, param in sharded_params.items():
                # Both parameter types implement .gather() and .free()
                gathered_tensor = param.gather().data.clone()

                if dist.get_rank() == 0:
                    state_dict[name] = gathered_tensor

                param.free()

        return state_dict

    @staticmethod
    def load_state_dict(
        sharded_params: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> None:
        """
        Loads a state dict back into either ShardedParameter or DTensorShardedParameter.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if state_dict_type == StateDictType.LOCAL:
            for name, param in sharded_params.items():
                if name in state_dict:
                    if hasattr(param, "dtensor"):
                        # DTensor target
                        param.dtensor.to_local().data.copy_(state_dict[name])
                    else:
                        # Manual target
                        param.local_param.data.copy_(state_dict[name])

        elif state_dict_type == StateDictType.FULL:
            for name, param in sharded_params.items():
                # 1. Prepare global tensor buffer across all ranks
                if rank == 0:
                    if name not in state_dict:
                        raise KeyError(f"Missing key '{name}' in full state dict provided to Rank 0.")
                    full_tensor = state_dict[name].clone()
                else:
                    full_tensor = torch.empty(param.global_shape)

                # 2. Broadcast full tensor from Rank 0 to all other ranks
                dist.broadcast(full_tensor, src=0)

                # 3. Branching based on implementation class:
                if hasattr(param, "dtensor"):
                    # --- DTensor Path ---
                    # DTensor handles its own sharding across the DeviceMesh
                    new_dtensor = distribute_tensor(
                        full_tensor, device_mesh=param.device_mesh, placements=[Shard(dim=0)]
                    )
                    param.dtensor = new_dtensor
                    param.free()

                else:
                    # --- Manual ShardedParameter Path ---
                    flat_param = full_tensor.flatten()

                    padding_needed = getattr(param, "padding_needed", 0)
                    if padding_needed > 0:
                        flat_param = torch.cat([flat_param, torch.zeros(padding_needed)])

                    chunks = torch.chunk(flat_param, world_size)
                    rank_chunk = chunks[rank]

                    # Fix shape mismatch: Reshape 1D flat chunk to match 2D/nD local_param shape
                    reshaped_chunk = rank_chunk.view(param.local_param.shape)
                    param.local_param.data.copy_(reshaped_chunk)
                    param.free()

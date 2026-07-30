from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from gloofsdp.collectives import all_gather_parameter
from gloofsdp.hooks import BackwardHook
from gloofsdp.memory_tracker import memory_tracker


class ShardedParameter:
    def __init__(self, name: str, module: nn.Module, param: nn.Parameter, world_size: int, rank: int):
        self.name = name
        self.module = module
        self.parameter_name = name.split(".")[-1]
        self.global_shape = param.shape
        self.world_size = world_size
        self.rank = rank

        # Flatten and calculate sharding ranges
        flat_param = param.data.flatten()
        self.padding_needed = (world_size - (flat_param.numel() % world_size)) % world_size
        if self.padding_needed > 0:
            flat_param = torch.cat([flat_param, torch.zeros(self.padding_needed, device=flat_param.device)])

        self.shard_size = flat_param.numel() // world_size
        start_idx = rank * self.shard_size
        end_idx = start_idx + self.shard_size
        self.shard_range = (start_idx, end_idx)

        # Local shard parameter (tracked by the optimizer)
        chunks = torch.chunk(flat_param, world_size)
        local_shard_data = chunks[rank].clone()
        self.local_param = nn.Parameter(local_shard_data, requires_grad=param.requires_grad)

        # Track persistent local shard memory allocation
        memory_tracker.track_sharded_param(self.local_param)

        # Temporary full tensor during forward/backward
        self.full_param: Optional[nn.Parameter] = None

    def gather(self) -> nn.Parameter:
        """All-gathers shards and attaches full parameter to module."""
        # Always gather fresh data from local_param (which optimizer updated)
        full_tensor_data = all_gather_parameter(self.local_param.data, self.global_shape)

        self.full_param = nn.Parameter(full_tensor_data, requires_grad=self.local_param.requires_grad)

        # Track newly materialized gathered parameter
        memory_tracker.track_gathered_param(self.full_param)

        if self.full_param.requires_grad:
            # Register explicit BackwardHook instance
            self.full_param.register_hook(BackwardHook(self))

        self._set_on_module(self.full_param)
        return self.full_param

    def free(self) -> None:
        """Re-attaches local_param to the module and discards full_param."""
        if self.full_param is not None:
            # Untrack full parameter allocation
            memory_tracker.untrack_gathered_param(self.full_param)

        self._set_on_module(self.local_param)
        self.full_param = None

    def _set_on_module(self, tensor: nn.Parameter) -> None:
        # self.module is already the parent leaf module (e.g. Linear object)
        # self.parameter_name is just "weight" or "bias"
        attr_name = self.parameter_name

        if hasattr(self.module, attr_name):
            delattr(self.module, attr_name)
        setattr(self.module, attr_name, tensor)

    # def _set_on_module(self, tensor: nn.Parameter) -> None:
    #     attrs = self.name.split(".")
    #     obj = self.module
    #     for attr in attrs[:-1]:
    #         obj = getattr(obj, attr)

    #     # Safely remove existing attribute before setting new Parameter
    #     if hasattr(obj, attrs[-1]):
    #         delattr(obj, attrs[-1])
    #     setattr(obj, attrs[-1], tensor)


class DTensorShardedParameter:
    def __init__(
        self,
        name: str,
        module: nn.Module,
        param: nn.Parameter,
        device_mesh: DeviceMesh,
    ) -> None:
        self.name = name
        self.parameter_name = name.split(".")[-1]
        self.module = module
        self.device_mesh = device_mesh
        self.global_shape = param.shape

        # Convert the full global parameter into a DTensor sharded along dim 0
        # This replaces our manual padding and chunking logic!
        with torch.no_grad():
            self.dtensor: DTensor = distribute_tensor(
                param.data, device_mesh=self.device_mesh, placements=[Shard(dim=0)]
            )

        self.shard_size = self.dtensor.numel() // self.device_mesh.size()
        start_idx = self.device_mesh.get_rank() * self.shard_size
        end_idx = start_idx + self.shard_size
        self.shard_range = (start_idx, end_idx)

        self.local_param = self.dtensor.to_local()

        # Replace the module parameter with the local tensor shard
        self._set_on_module(nn.Parameter(self.dtensor.to_local()))

    def gather(self) -> torch.Tensor:
        """
        Redistributes the sharded DTensor to a Replicated state (All-Gather).
        """
        # Replicate the DTensor across the mesh
        replicated_dtensor = self.dtensor.redistribute(device_mesh=self.device_mesh, placements=[Replicate()])

        # Unwrap to the raw local PyTorch tensor (full global shape)
        full_tensor = replicated_dtensor.to_local()
        self._set_on_module(nn.Parameter(full_tensor))
        return full_tensor

    def free(self) -> None:
        """
        Reverts module attribute back to the local shard to free full-size memory.
        """
        self._set_on_module(nn.Parameter(self.dtensor.to_local()))

    def _set_on_module(self, tensor: nn.Parameter) -> None:
        if hasattr(self.module, self.parameter_name):
            delattr(self.module, self.parameter_name)
        setattr(self.module, self.parameter_name, tensor)

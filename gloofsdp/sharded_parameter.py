import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from gloofsdp.memory_tracker import memory_tracker


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

        # 1. Create local parameter shard directly (this is what the optimizer will update)
        # TODO (andrei aksionau): if the optimizer will update only the local shard, then
        # why in the backward hook it gets the full gradient?
        with torch.no_grad():
            temp_dtensor = distribute_tensor(param.data, device_mesh=self.device_mesh, placements=[Shard(dim=0)])
            # Make sure local_param is a leaf nn.Parameter!
            self.local_param = nn.Parameter(temp_dtensor.to_local().clone())

        # TODO (andrei aksionau): this is called in wrapper too. Need to decide where to keep
        #   and I think it should be here. ⬇
        # Set the local shard on the module initially
        self.free()

        # TODO (andrei aksionau): why do I need this? Feels like it's for memory tracking or
        # for visualizer
        # self.shard_size = self.dtensor.numel() // self.device_mesh.size()
        self.shard_size = self.local_param.numel() // self.device_mesh.size()
        start_idx = self.device_mesh.get_rank() * self.shard_size
        end_idx = start_idx + self.shard_size
        self.shard_range = (start_idx, end_idx)

        # Track persistent local shard memory allocation
        memory_tracker.track_sharded_param(self.local_param)

        # Replace the module parameter with the local tensor shard
        # self._set_on_module(nn.Parameter(self.dtensor.to_local()))
        # self._set_on_module(self.local_param)

    def gather(self) -> torch.Tensor:
        """
        Reconstructs DTensor from current local_param, redistributes to Replicated,
        and sets the gathered parameter on the module.
        """

        # TODO (andrei aksionau): this doesn't look pretty - here we explicitly call
        # redistribute, but for the backward we use the hook.
        # Or it is the only way to do it?

        # 1. Wrap the CURRENT (potentially optimizer-updated) local_param as a Sharded DTensor
        sharded_dtensor = DTensor.from_local(self.local_param, device_mesh=self.device_mesh, placements=[Shard(dim=0)])

        # 2. Gather / Redistribute across mesh -> Replicate()
        replicated_dtensor = sharded_dtensor.redistribute(device_mesh=self.device_mesh, placements=[Replicate()])

        # Track newly materialized gathered parameter
        memory_tracker.track_gathered_param(replicated_dtensor)

        # 3. Unwrap local tensor and attach to module as a Parameter requiring grad
        # TODO (andrei aksionau): It might be used not only for training, so grad
        #   requirement shouldn't be hard-coded
        full_tensor = replicated_dtensor.to_local().detach().requires_grad_(True)
        self._set_on_module(full_tensor)

        return full_tensor

    def gather1(self) -> torch.Tensor:
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
        # if self.full_param is not None:
        #     # Untrack full parameter allocation
        #     memory_tracker.untrack_gathered_param(self.full_param)
        # self._set_on_module(nn.Parameter(self.dtensor.to_local()))
        self._set_on_module(self.local_param)

    def _set_on_module(self, tensor: nn.Parameter) -> None:
        if hasattr(self.module, self.parameter_name):
            delattr(self.module, self.parameter_name)
        setattr(self.module, self.parameter_name, tensor)

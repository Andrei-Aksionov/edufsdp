import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from gloofsdp.memory_tracker import memory_tracker


class DTensorShardedParameter:
    """
    Manages the lifecycle of a sharded parameter under FSDP.

    Responsible for:
    1. Splitting the full parameter into a rank-local shard owned by the optimizer.
    2. All-Gathering the full parameter tensor on-demand during forward/backward steps.
    3. Offloading/freeing full parameter memory after computation completes.
    """

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

        # 1. Shard initial weight tensor across dim 0 using DTensor primitives.
        # local_param is a leaf nn.Parameter that the optimizer interacts with directly.
        with torch.no_grad():
            temp_dtensor = distribute_tensor(param.data, device_mesh=self.device_mesh, placements=[Shard(dim=0)])
            self.local_param = nn.Parameter(
                temp_dtensor.to_local().clone(),
                requires_grad=param.requires_grad,
            )

        # 2. Track element ranges for memory accounting and debug visualization.
        self.shard_size = self.local_param.numel()
        start_idx = self.device_mesh.get_rank() * self.shard_size
        end_idx = start_idx + self.shard_size
        self.shard_range = (start_idx, end_idx)

        # Register shard with global memory tracking utility
        memory_tracker.track_sharded_param(self.local_param)

        # 3. Immediately replace module parameter with local shard to save memory.
        self.free()

    def gather(self) -> torch.Tensor:
        """
        All-Gathers local parameter shards across the device mesh to reconstruct
        the full parameter, attaching it to the target module.
        """
        # Wrap local tensor slice back into a DTensor with Shard(0) placement
        sharded_dtensor = DTensor.from_local(
            self.local_param,
            device_mesh=self.device_mesh,
            placements=[Shard(dim=0)],
        )

        # Perform All-Gather collective across device mesh (Shard(0) -> Replicate())
        replicated_dtensor = sharded_dtensor.redistribute(device_mesh=self.device_mesh, placements=[Replicate()])

        # Log gathered state in memory profiler
        memory_tracker.track_gathered_param(replicated_dtensor)

        # Extract underlying local storage.
        # Inherit requires_grad status from self.local_param unless grad evaluation is globally disabled.
        requires_grad = self.local_param.requires_grad and torch.is_grad_enabled()
        full_tensor = replicated_dtensor.to_local().detach().requires_grad_(requires_grad)

        # Swap gathered tensor onto module attribute so forward pass can use it
        self._set_on_module(full_tensor)
        return full_tensor

    def free(self) -> None:
        """
        Restores the module attribute to point back to the smaller local shard,
        allowing the garbage collector to release the full tensor memory.
        """
        self._set_on_module(self.local_param)

    def _set_on_module(self, tensor: torch.Tensor) -> None:
        """Helper to replace module attribute cleanly without PyTorch parameter name collision errors."""
        if hasattr(self.module, self.parameter_name):
            delattr(self.module, self.parameter_name)
        setattr(self.module, self.parameter_name, tensor)

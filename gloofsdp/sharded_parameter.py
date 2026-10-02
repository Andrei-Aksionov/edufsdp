"""
Parameter sharding abstraction for Educational FSDP.

Provides `DTensorShardedParameter` to wrap standard PyTorch parameters, managing:
1. Sharding original weight tensors across a distributed DeviceMesh.
2. Symmetric state transitions (`set_unsharded` / `set_sharded`).
3. Memory cleanup and attribute lifecycle state.
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor


class ShardedParameter:
    """
    Pure data container managing parameter shard metadata and tensor state on a Module.

    Responsible for:
    1. Distributing the raw parameter into rank-local shards.
    2. Exposing state mutation methods (`set_unsharded`, `set_sharded`) without executing
       autograd or collective logic directly.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        param: nn.Parameter,
        device_mesh: DeviceMesh,
    ) -> None:
        """
        Initializes parameter sharding state.

        Args:
            name: Fully qualified parameter name (e.g., 'layer1.0.weight').
            module: Parent `nn.Module` container that owns this parameter attribute.
            param: Unsharded source `nn.Parameter` instance.
            device_mesh: Distributed DeviceMesh defining rank layout.
        """
        self.name = name
        self.parameter_name = name.split(".")[-1]
        self.module = module
        self.device_mesh = device_mesh
        self.global_shape = param.shape

        # 1. Convert initial tensor to DTensor and wrap as a leaf nn.Parameter.
        #
        # WHY torch.no_grad() IS USED HERE:
        # - Prevents PyTorch from recording `distribute_tensor` ops in the global autograd graph
        #   during setup, which avoids memory leaks before training begins.
        # - Ensures `sharded_dtensor` is instantiated without autograd history (`grad_fn=None`),
        #   allowing PyTorch to construct a valid, clean leaf `nn.Parameter`.
        with torch.no_grad():
            sharded_dtensor = distribute_tensor(param.data, device_mesh=self.device_mesh, placements=[Shard(dim=0)])
            self.sharded_param = nn.Parameter(
                sharded_dtensor,
                requires_grad=param.requires_grad,
            )

        # 2. Mount the DTensor Parameter directly on the target module attribute
        setattr(self.module, self.parameter_name, self.sharded_param)

        # 3. Local storage metadata
        self.shard_size = self.sharded_param.to_local().numel()
        start_idx = self.device_mesh.get_rank() * self.shard_size
        end_idx = start_idx + self.shard_size
        self.shard_range = (start_idx, end_idx)

        # Storage reference when parameter is gathered
        self._full_tensor: torch.Tensor | None = None

    @property
    def local_shard(self) -> torch.Tensor:
        """Returns the underlying rank-local raw storage tensor."""
        return self.sharded_param.to_local()

    @property
    def is_gathered(self) -> bool:
        """Returns True if the full parameter is currently attached to the module."""
        return self._full_tensor is not None

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the currently active tensor (gathered or sharded)."""
        if self.is_gathered and self._full_tensor is not None:
            return self._full_tensor.shape
        return self.local_shard.shape

    def set_unsharded(self, full_tensor: torch.Tensor) -> None:
        """
        Attaches the full gathered tensor to the module attribute.

        Bypasses `nn.Module.__setattr__` checks so PyTorch allows plain `torch.Tensor`
        assignment during computation without breaking autograd tracking.
        """
        self._full_tensor = full_tensor
        self.module.__dict__[self.parameter_name] = full_tensor

    def set_sharded(self) -> None:
        """
        Restores module attribute back to the local DTensor shard, allowing
        garbage collection of full tensor memory.
        """
        self.module.__dict__[self.parameter_name] = self.sharded_param
        self._full_tensor = None

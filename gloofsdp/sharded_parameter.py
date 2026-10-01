"""
Parameter sharding abstraction for Educational FSDP.

Provides `DTensorShardedParameter` to wrap standard PyTorch parameters, managing:
1. Sharding original weight tensors across a distributed DeviceMesh.
2. Dynamic parameter reconstruction (`gather`) via `FSDPAllGatherFunction`.
3. Memory release (`free`) back to rank-local parameter shards.
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from gloofsdp.hooks import FSDPAllGatherFunction


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

        # 1. Convert initial tensor to DTensor and wrap as a leaf nn.Parameter
        with torch.no_grad():
            sharded_dtensor = distribute_tensor(param.data, device_mesh=self.device_mesh, placements=[Shard(dim=0)])
            # This parameter stays persistently assigned to self.module
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

        # Handle temporary storage during forward/backward execution
        self._full_tensor: torch.Tensor | None = None

    @property
    def is_gathered(self) -> bool:
        return self._full_tensor is not None

    @property
    def shape(self) -> torch.Size:
        if self.is_gathered:
            return self._full_tensor.shape
        return self.sharded_param.to_local().shape

    def gather(self) -> torch.Tensor:
        """
        All-Gathers local parameter shards and attaches full tensor onto parent module.

        Executes `FSDPAllGatherFunction.apply()`, which registers the autograd node
        required for automatic Reduce-Scatter execution during backward passes.

        Returns:
            torch.Tensor: Unwrapped full parameter tensor attached to the module.
        """

        if self.is_gathered:
            return self._full_tensor

        # Get local storage tensor for collective operation
        local_shard = self.sharded_param.to_local()

        if self.sharded_param.requires_grad and torch.is_grad_enabled():
            full_tensor = FSDPAllGatherFunction.apply(local_shard, self)
        else:
            with torch.no_grad():
                full_tensor = FSDPAllGatherFunction.apply(local_shard, self)

        self._full_tensor = full_tensor

        # Temporarily replace the module attribute with the gathered full tensor
        # so F.linear receives regular torch.Tensors for both input and weight
        # Bypasses Module.__setattr__ check so PyTorch allows plain Tensor
        self.module.__dict__[self.parameter_name] = full_tensor

        return full_tensor

    def free(self) -> None:
        """
        Restores module attribute back to local shard, allowing garbage collection of full tensor.
        """
        # Restore the sharded DTensor Parameter back on the module
        self.module.__dict__[self.parameter_name] = self.sharded_param
        self._full_tensor = None

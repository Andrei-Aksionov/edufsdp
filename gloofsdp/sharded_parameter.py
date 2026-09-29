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

        # 1. Shard original parameter along dimension 0 across the device mesh
        with torch.no_grad():
            temp_dtensor = distribute_tensor(param.data, device_mesh=self.device_mesh, placements=[Shard(dim=0)])
            # Create leaf parameter managed directly by the optimizer
            # TODO: bad naming - have to always call .to_local() on a var that is called local_param
            self.local_param = nn.Parameter(
                temp_dtensor,
                requires_grad=param.requires_grad,
            )

        # 2. Track element ranges for logging and visual profiling
        # local_param is ALREADY sharded, so local_param.numel() is local shard size
        self.shard_size = self.local_param.numel()
        start_idx = self.device_mesh.get_rank() * self.shard_size
        end_idx = start_idx + self.shard_size
        self.shard_range = (start_idx, end_idx)

        # Track sharded memory allocation
        memory_tracker.track_sharded_param(self.local_param)

        # 3. Immediately set module attribute to initial local shard
        self.free()

    def gather(self) -> torch.Tensor:
        """
        All-Gathers local parameter shards and attaches full tensor onto parent module.

        Executes `FSDPAllGatherFunction.apply()`, which registers the autograd node
        required for automatic Reduce-Scatter execution during backward passes.

        Returns:
            torch.Tensor: Unwrapped full parameter tensor attached to the module.
        """

        if self.local_param.requires_grad and torch.is_grad_enabled():
            # Apply custom autograd function to register node in autograd graph
            full_tensor = FSDPAllGatherFunction.apply(self.local_param.to_local(), self)
        else:
            # Inference/no-grad evaluation path
            with torch.no_grad():
                full_tensor = FSDPAllGatherFunction.apply(self.local_param.to_local(), self)

        # Assign reconstructed tensor onto module attribute for forward computation
        self._set_on_module(full_tensor)
        return full_tensor

    def free(self) -> None:
        """
        Restores module attribute back to local shard, allowing garbage collection of full tensor.
        """
        self._set_on_module(self.local_param)

    def _set_on_module(self, tensor: torch.Tensor) -> None:
        """Helper method to update module attribute without parameter collision errors."""
        if hasattr(self.module, self.parameter_name):
            delattr(self.module, self.parameter_name)
        setattr(self.module, self.parameter_name, tensor)

"""
Parameter sharding abstraction for Educational FSDP.

Provides `ShardedParameter` to wrap standard PyTorch `nn.Parameter` instances,
managing:
  1. Sharding original weight tensors across a distributed `DeviceMesh`.
  2. Symmetric state transitions (`set_unsharded` <-> `set_sharded`).
  3. Memory cleanup and local vs. full tensor attribute lifecycle.

Parameter Lifecycle State Machine:

                 ┌──────────────────────────────────────────────┐
                 │                SHARDED STATE                 │
                 │  module.weight/bias = DTensor (1/N memory)   │
                 └──────────────────┬───────────────────────────┘
                                    │       ▲
                    set_unsharded() │       │ set_sharded()
                                    ▼       │
                 ┌──────────────────────────┴───────────────────┐
                 │               GATHERED STATE                 │
                 │  module.weight/bias = Tensor (Full memory)   │
                 └──────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor


class ShardedParameter:
    """
    Manages parameter shard metadata and tensor state transitions on a target `nn.Module`.

    This class acts as a bridge between PyTorch's `DTensor` sharding primitives and the
    underlying `nn.Module` execution graph.

    Key Responsibilities:
      1. Shards full parameter data across a `DeviceMesh` along dimension 0 (`Shard(dim=0)`).
      2. Dynamically swaps module attributes between local `DTensor` shards and full
         unsharded `torch.Tensor` instances during forward/backward steps.
      3. Provides memory-tracking properties (local shard sizes, global ranges).
    """

    def __init__(
        self,
        fqn: str,
        module: nn.Module,
        param: nn.Parameter,
        device_mesh: DeviceMesh,
    ) -> None:
        """
        Initializes parameter sharding state and installs the sharded DTensor onto the module.

        Args:
            fqn (str): Fully qualified parameter name (e.g., 'layer1.weight').
            module (nn.Module): Parent PyTorch module owning this parameter.
            param (nn.Parameter): Unsharded source parameter to split across ranks.
            device_mesh (DeviceMesh): PyTorch DeviceMesh defining rank layout.
        """
        self.fqn = fqn
        self.parameter_name = fqn.split(".")[-1]
        self.module = module
        self.device_mesh = device_mesh

        # ---------------------------------------------------------------------
        # 1. Distribute raw tensor memory into a DTensor parameter
        #
        # Why torch.no_grad() is required here:
        # - Prevents PyTorch from recording `distribute_tensor` ops in the global
        #   autograd graph during model initialization (which causes memory leaks).
        # - Ensures `sharded_dtensor` is created as a clean leaf node (grad_fn=None).
        # ---------------------------------------------------------------------
        with torch.no_grad():
            sharded_dtensor = distribute_tensor(param.data, device_mesh=self.device_mesh, placements=[Shard(dim=0)])
            self.dtensor = nn.Parameter(
                sharded_dtensor,
                requires_grad=param.requires_grad,
            )

        # 2. Mount the DTensor Parameter directly on the target module attribute
        setattr(self.module, self.parameter_name, self.dtensor)

        # Reference holder for the full gathered tensor during execution
        self._gathered_tensor: torch.Tensor | None = None

    @property
    def local_shard(self) -> torch.Tensor:
        """
        Returns the raw, rank-local storage slice (`torch.Tensor`) backing the DTensor.
        """
        return self.dtensor.to_local()

    @property
    def is_gathered(self) -> bool:
        """
        `True` if the full unsharded tensor is currently attached to the parent module.
        """
        return self._gathered_tensor is not None

    @property
    def shape(self) -> torch.Size:
        """
        Returns the active tensor shape depending on lifecycle state:
          - GATHERED: Global shape (e.g., [24, 18])
          - SHARDED: Local shard shape (e.g., [12, 18] on 2 ranks)
        """
        return self._gathered_tensor.shape if self.is_gathered else self.local_shard.shape

    def set_unsharded(self, full_tensor: torch.Tensor) -> None:
        """
        Attaches the full gathered tensor to the parent module attribute.

        Standard `setattr(module, param_name, full_tensor)` invokes PyTorch's
        `nn.Module.__setattr__`, which rejects assigning a plain `torch.Tensor`
        to an existing `nn.Parameter` attribute.

        By assigning directly into `module.__dict__`, we bypass parameter type checks
        so native C++ PyTorch functions (e.g., `F.linear`) can directly consume standard
        contiguous tensors during forward and backward computations.

        Args:
            full_tensor (torch.Tensor): Unsharded full-size tensor gathered across ranks.
        """
        self._gathered_tensor = full_tensor
        self.module.__dict__[self.parameter_name] = full_tensor

    def set_sharded(self) -> None:
        """
        Restores the parent module attribute back to the rank-local DTensor `nn.Parameter`.

        This releases references to the full unsharded tensor memory, allowing
        PyTorch's CUDA memory manager / Python garbage collector to reclaim the memory.
        """
        self.module.__dict__[self.parameter_name] = self.dtensor
        self._gathered_tensor = None

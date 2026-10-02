"""
Autograd and module execution hooks for Educational FSDP.

Provides:
1. `FSDPAllGatherFunction`: Standard `torch.autograd.Function` handling All-Gather and Reduce-Scatter.
2. `ForwardPreHook`: Module pre-hook managing parameter reconstruction before calculation.
3. `ForwardPostHook`: Module post-hook clearing full weights back to local shards after forward pass.
4. `BackwardPreHook` / `BackwardPostHook`: Module backward hooks managing parameter lifecycle during backward pass.


========================================================================================
                                    FORWARD PASS
========================================================================================

 [ Forward Pre-Hook ] ──► All-Gather #1 ──► Module Computation ──► [ Forward Post-Hook ]
 (Gathers Full Weight)                       (using Full Weight)   (Frees Full Weight)

========================================================================================
                                    BACKWARD PASS
========================================================================================

 [ Autograd reaches Layer ]
           │
           ▼
 [ Pre-Backward Hook / Re-Gather ]
           │
           ├──► Executes All-Gather #2 (Re-creates Full Weight W)
           │
           ▼
 [ Module Matrix Math ]
           │
           ├──► Calculates W = X^T @ Y  (Creates `grad_output`)
           │
           ▼
 [ FSDPAllGatherFunction.backward(grad_output) ]
           │
           ├──► 1. Takes `grad_output` (Full Gradient)
           ├──► 2. Runs Reduce-Scatter ──► Produces `sharded_grad`
           ├──► 3. Assigns `sharded_grad` to `local_param.grad`
           └──► 4. Frees Full Weight W again
"""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from gloofsdp.logging import logger

if TYPE_CHECKING:
    from gloofsdp.sharded_parameter import DTensorShardedParameter


class FSDPAllGatherFunction(torch.autograd.Function):
    """
    Symmetric autograd function pairing forward parameter All-Gather with backward gradient Reduce-Scatter.
    """

    @staticmethod
    def forward(ctx: Any, param: "DTensorShardedParameter") -> torch.Tensor:
        """
        Gathers local parameter shards across ranks to form the full global parameter.
        """
        ctx.param = param

        # 1. Wrap rank-local parameter shard as a DTensor sharded along dimension 0
        sharded_dtensor = DTensor.from_local(
            param.local_shard,
            device_mesh=param.device_mesh,
            placements=[Shard(dim=0)],
        )

        # 2. Execute All-Gather collective across process group (Shard(0) -> Replicate())
        replicated_dtensor = sharded_dtensor.redistribute(
            device_mesh=param.device_mesh,
            placements=[Replicate()],
        )

        result = replicated_dtensor.to_local()

        logger.log_transition(
            event_name="all_gather",
            param_name=param.name,
            total_elements=param.global_shape.numel(),
            shard_range=param.shard_range,
            before_gathered=False,
            after_gathered=True,
            shape=f"{tuple(param.local_shard.shape)} -> {tuple(result.shape)}",
        )

        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        """
        Reduces incoming full-shape gradient across ranks back into a local gradient shard.
        """
        param: DTensorShardedParameter = ctx.param

        # 1. Wrap un-reduced full gradient as Partial() across the process group
        grad_dtensor = DTensor.from_local(
            grad_output,
            device_mesh=param.device_mesh,
            placements=[Partial()],
        )

        # 2. Execute Reduce-Scatter collective across mesh (Partial() -> Shard(0))
        sharded_grad_dtensor = grad_dtensor.redistribute(
            device_mesh=param.device_mesh,
            placements=[Shard(dim=0)],
        )

        sharded_grad = sharded_grad_dtensor.to_local()

        # 3. Average reduced additive gradient sum by process group world size
        world_size = param.device_mesh.size()
        sharded_grad = sharded_grad / world_size

        logger.log_transition(
            event_name="reduce_scatter",
            param_name=f"{param.name}.grad",
            total_elements=param.global_shape.numel(),
            shard_range=param.shard_range,
            before_gathered=True,
            after_gathered=False,
            shape=f"{tuple(grad_output.shape)} -> {tuple(sharded_grad.shape)}",
        )

        # Matches forward inputs: (local_shard, device_mesh, param_name, shard_range, global_numel)
        return sharded_grad


class ForwardPreHook:
    """Module pre-forward hook that gathers parameter shards before computation starts."""

    def __init__(self, sharded_param: "DTensorShardedParameter"):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        if self.sharded_param.is_gathered:
            return

        # Run All-Gather with or without autograd graph tracking
        if self.sharded_param.sharded_param.requires_grad and torch.is_grad_enabled():
            full_tensor = FSDPAllGatherFunction.apply(self.sharded_param)
        else:
            with torch.no_grad():
                full_tensor = FSDPAllGatherFunction.apply(self.sharded_param)

        self.sharded_param.set_unsharded(full_tensor)


class ForwardPostHook:
    """Module post-forward hook that releases full parameters immediately after computation finishes."""

    def __init__(self, sharded_param: "DTensorShardedParameter"):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        before_shape = self.sharded_param.shape
        self.sharded_param.set_sharded()
        after_shape = self.sharded_param.shape

        logger.log_transition(
            event_name="free",
            param_name=self.sharded_param.name,
            total_elements=self.sharded_param.global_shape.numel(),
            shard_range=self.sharded_param.shard_range,
            before_gathered=True,
            after_gathered=False,
            shape=f"{tuple(before_shape)} -> {tuple(after_shape)}",
        )


class BackwardPreHook:
    """Module backward pre-hook that re-gathers full weights for gradient calculations."""

    def __init__(self, sharded_param: "DTensorShardedParameter"):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, grad_output: Any) -> None:
        if self.sharded_param.is_gathered:
            return

        # WHY torch.no_grad() IS USED HERE:
        # - The primary autograd graph node for `FSDPAllGatherFunction` was recorded during forward execution.
        # - Omitting `torch.no_grad()` here would create a duplicate AllGather node inside the backward graph
        #   while the autograd engine is actively executing, causing runtime graph mutation errors.
        with torch.no_grad():
            full_tensor = FSDPAllGatherFunction.apply(self.sharded_param)

        self.sharded_param.set_unsharded(full_tensor)


class BackwardPostHook:
    """Module backward post-hook that frees the gathered weight after gradient computations."""

    def __init__(self, sharded_param: "DTensorShardedParameter") -> None:
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        before_shape = self.sharded_param.shape
        self.sharded_param.set_sharded()
        after_shape = self.sharded_param.shape

        logger.log_transition(
            event_name="free",
            param_name=self.sharded_param.name,
            total_elements=self.sharded_param.global_shape.numel(),
            shard_range=self.sharded_param.shard_range,
            before_gathered=True,
            after_gathered=False,
            shape=f"{tuple(before_shape)} -> {tuple(after_shape)}",
        )

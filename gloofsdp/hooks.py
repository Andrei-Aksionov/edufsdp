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
    from gloofsdp.sharded_parameter import ShardedParameter


class FSDPAllGatherFunction(torch.autograd.Function):
    """
    Symmetric autograd function pairing forward parameter All-Gather with backward gradient Reduce-Scatter.
    """

    @staticmethod
    def forward(ctx: Any, local_param: torch.Tensor, sharded_param: "ShardedParameter") -> torch.Tensor:
        """Gathers rank-local parameter shards into the full global parameter.

        By receiving ``local_param`` explicitly as a tensor argument alongside the
        ``sharded_param`` metadata container, PyTorch autograd can directly track
        the leaf tensor in the execution graph, ensuring backward gradients are
        correctly routed back to ``local_param.grad``.

        Args:
            ctx: Autograd context used to store metadata across the forward and
                backward passes.
            local_param: The rank-local slice of the parameter tensor carrying
                autograd graph history (`requires_grad=True`).
            sharded_param: Parameter container holding distributed topology,
                device mesh, and sharding metadata.

        Returns:
            torch.Tensor: The reconstructed global parameter tensor assembled
            via All-Gather across the device mesh.
        """

        ctx.param = sharded_param

        # 1. Wrap rank-local parameter shard as a DTensor sharded along dimension 0
        sharded_dtensor = DTensor.from_local(
            local_param,
            device_mesh=sharded_param.device_mesh,
            placements=[Shard(dim=0)],
        )

        # 2. Execute All-Gather collective across process group (Shard(0) -> Replicate())
        replicated_dtensor = sharded_dtensor.redistribute(
            device_mesh=sharded_param.device_mesh,
            placements=[Replicate()],
        ).to_local()

        logger.log_transition(
            event_name="all_gather",
            parameter_name=sharded_param.fqn,
            size_before=sharded_param.local_shard.shape,
            size_after=replicated_dtensor.shape,
        )

        return replicated_dtensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Reduces the incoming global gradient back into a local gradient shard.

        Applies a Reduce-Scatter collective across the process group to aggregate and
        shard the accumulated global gradient. The resulting local gradient shard is
        returned as the first element of the tuple to match ``local_param`` in the
        forward signature, enabling PyTorch autograd to assign it to ``local_param.grad``.

        Args:
            ctx: Autograd context containing stored distributed metadata (device mesh,
                sharding dimensions) from the forward pass.
            grad_output: Incoming full-shape gradient tensor computed during the backward
                pass for the gathered parameter.

        Returns:
            tuple[torch.Tensor, None]: A 2-tuple where:
                - The first element is the rank-local sharded gradient tensor for
                  ``local_param``, scaled by world size.
                - The second element is ``None``, matching non-tensor metadata argument
                  ``sharded_param`` from the forward pass.
        """

        param: ShardedParameter = ctx.param

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
            parameter_name=f"{param.fqn}.grad",
            size_before=grad_output.shape,
            size_after=sharded_grad.shape,
        )

        # Matches forward inputs
        return sharded_grad, None


class ForwardPreHook:
    """Module pre-forward hook that gathers parameter shards before computation starts."""

    def __init__(self, sharded_param: "ShardedParameter"):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        if self.sharded_param.is_gathered:
            return

        # Run All-Gather with or without autograd graph tracking
        if self.sharded_param.dtensor.requires_grad and torch.is_grad_enabled():
            full_tensor = FSDPAllGatherFunction.apply(self.sharded_param.local_shard, self.sharded_param)
        else:
            with torch.no_grad():
                full_tensor = FSDPAllGatherFunction.apply(self.sharded_param.local_shard, self.sharded_param)

        self.sharded_param.set_unsharded(full_tensor)


class ForwardPostHook:
    """Module post-forward hook that releases full parameters immediately after computation finishes."""

    def __init__(self, sharded_param: "ShardedParameter"):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        before_shape = self.sharded_param.shape
        self.sharded_param.set_sharded()
        after_shape = self.sharded_param.shape

        logger.log_transition(
            event_name="free",
            parameter_name=self.sharded_param.fqn,
            size_before=before_shape,
            size_after=after_shape,
        )


class BackwardPreHook:
    """Module backward pre-hook that re-gathers full weights for gradient calculations."""

    def __init__(self, sharded_param: "ShardedParameter"):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, grad_output: Any) -> None:
        if self.sharded_param.is_gathered:
            return

        # WHY torch.no_grad() IS USED HERE:
        # - The primary autograd graph node for `FSDPAllGatherFunction` was recorded during forward execution.
        # - Omitting `torch.no_grad()` here would create a duplicate AllGather node inside the backward graph
        #   while the autograd engine is actively executing, causing runtime graph mutation errors.
        with torch.no_grad():
            full_tensor = FSDPAllGatherFunction.apply(self.sharded_param.local_shard, self.sharded_param)

        self.sharded_param.set_unsharded(full_tensor)


class BackwardPostHook:
    """Module backward post-hook that frees the gathered weight after gradient computations."""

    def __init__(self, sharded_param: "ShardedParameter") -> None:
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        before_shape = self.sharded_param.shape
        self.sharded_param.set_sharded()
        after_shape = self.sharded_param.shape

        logger.log_transition(
            event_name="free",
            parameter_name=self.sharded_param.fqn,
            size_before=before_shape,
            size_after=after_shape,
        )

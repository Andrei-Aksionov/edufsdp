"""
Autograd and module execution hooks for Educational FSDP.

This module provides:
1. `FSDPAllGatherFunction`: A symmetric `torch.autograd.Function` bridging forward All-Gather
   and backward Reduce-Scatter operations inside PyTorch's dynamic autograd graph.
2. `ForwardPreHook`: A PyTorch module hook that reconstructs full weights before module computation.
3. `ForwardPostHook`: A PyTorch module hook that releases full weights after module computation.


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
from gloofsdp.memory_tracker import memory_tracker

if TYPE_CHECKING:
    from gloofsdp.sharded_parameter import DTensorShardedParameter


class FSDPAllGatherFunction(torch.autograd.Function):
    """
    Symmetric autograd function pairing forward parameter All-Gather with backward gradient Reduce-Scatter.

    By inheriting from `torch.autograd.Function`, PyTorch's engine automatically tracks
    this operation during forward execution and routes incoming gradients through `backward()`
    during loss.backward(). This eliminates the need for manual `register_hook()` calls on tensors.
    """

    @staticmethod
    def forward(
        ctx: Any,
        local_param: torch.Tensor,
        dtensor_param: "DTensorShardedParameter",
    ) -> torch.Tensor:
        """
        Gathers local parameter shards across ranks to form the full global parameter.

        Args:
            ctx: Autograd context object for storing state across forward/backward passes.
            local_param: Rank-local parameter shard tensor (leaf node managed by optimizer).
            dtensor_param: FSDP metadata container tracking mesh, shapes, and logging info.

        Returns:
            torch.Tensor: The reconstructed full parameter tensor allocated on the local process.
        """
        # Save metadata reference for backward execution pass
        ctx.dtensor_param = dtensor_param
        device_mesh = dtensor_param.device_mesh

        # Log parameter transition from Sharded to Gathered state
        logger.log_transition(
            event_name="all_gather",
            param_name=dtensor_param.name,
            total_elements=dtensor_param.global_shape.numel(),
            shard_range=dtensor_param.shard_range,
            before_gathered=False,
            after_gathered=True,
        )

        # 1. Wrap rank-local parameter shard as a DTensor sharded along dimension 0
        sharded_dtensor = DTensor.from_local(
            local_param,
            device_mesh=device_mesh,
            placements=[Shard(dim=0)],
        )

        # 2. Execute All-Gather collective across process group (Shard(0) -> Replicate())
        replicated_dtensor = sharded_dtensor.redistribute(
            device_mesh=device_mesh,
            placements=[Replicate()],
        )

        # Track full tensor allocation in global memory profiler
        memory_tracker.track_gathered_param(replicated_dtensor)

        # Return unwrapped raw PyTorch tensor to be set on module attribute
        return replicated_dtensor.to_local()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None]:
        """
        Reduces incoming full-shape gradient across ranks back into a local gradient shard.

        Triggered automatically by PyTorch's C++ autograd engine when reaching the
        <FSDPAllGatherFunctionBackward> node in the backward computation graph.

        Args:
            ctx: Autograd context containing saved references from `forward()`.
            grad_output: Full-shape gradient calculated with respect to the gathered full weight.

        Returns:
            tuple[torch.Tensor | None, None]: Gradient corresponding to `local_param` input,
            and `None` for the `dtensor_param` argument (which does not track gradients).
        """
        dtensor_param: "DTensorShardedParameter" = ctx.dtensor_param
        device_mesh = dtensor_param.device_mesh

        # Track full incoming gradient allocation
        memory_tracker.track_gradient(grad_output)

        logger.log_transition(
            event_name="reduce_scatter",
            param_name=f"grad({dtensor_param.name})",
            total_elements=dtensor_param.global_shape.numel(),
            shard_range=dtensor_param.shard_range,
            before_gathered=True,
            after_gathered=False,
        )

        # 1. Wrap un-reduced full gradient as Partial() across the process group
        # Partial() in DTensor.from_local() will force DTensor to communicate gradients
        # across processes during backward, bringing Rank 0 and Rank 1 back into alignment
        grad_dtensor = DTensor.from_local(
            grad_output,
            device_mesh=device_mesh,
            placements=[Partial()],
        )

        # 2. Execute Reduce-Scatter collective across mesh (Partial() -> Shard(0))
        sharded_grad_dtensor = grad_dtensor.redistribute(
            device_mesh=device_mesh,
            placements=[Shard(dim=0)],
        )

        sharded_grad = sharded_grad_dtensor.to_local()

        # 3. Average reduced additive gradient sum by process group world size
        world_size = device_mesh.size()
        sharded_grad = sharded_grad / world_size

        # Update global memory accounting
        memory_tracker.untrack_gradient(grad_output)
        memory_tracker.track_gradient(sharded_grad)

        # PyTorch autograd routes sharded_grad directly to local_param.grad
        return sharded_grad, None


class ForwardPreHook:
    """
    Module pre-forward hook that gathers parameter shards before computation starts.

    Registered on individual submodules via `module.register_forward_pre_hook()`.
    """

    def __init__(self, dtensor_param: "DTensorShardedParameter"):
        self.dtensor_param = dtensor_param

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        """Triggers All-Gather parameter reconstruction right before module.forward()."""
        self.dtensor_param.gather()


class ForwardPostHook:
    """
    Module post-forward hook that releases full parameters immediately after computation finishes.

    Registered on individual submodules via `module.register_forward_hook()`.
    """

    def __init__(self, dtensor_param: "DTensorShardedParameter"):
        self.dtensor_param = dtensor_param

    def __call__(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        """Restores module attribute back to local shard to release peak RAM."""
        logger.log_transition(
            event_name="free",
            param_name=self.dtensor_param.name,
            total_elements=self.dtensor_param.global_shape.numel(),
            shard_range=self.dtensor_param.shard_range,
            before_gathered=True,
            after_gathered=False,
        )
        self.dtensor_param.free()


class BackwardPreHook:
    """
    Module backward pre-hook that re-gathers full weights JUST BEFORE
    autograd computes the module's weight gradients.
    """

    def __init__(self, dtensor_param: "DTensorShardedParameter"):
        self.dtensor_param = dtensor_param

    def __call__(self, module: nn.Module, grad_output: Any) -> None:
        # All-Gather #2: Re-construct full weight tensor needed for dW math
        self.dtensor_param.gather()


class BackwardPostHook:
    def __init__(self, sharded_param: DTensorShardedParameter) -> None:
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        # Reshard weight data back to resting local shape
        self.sharded_param.free()

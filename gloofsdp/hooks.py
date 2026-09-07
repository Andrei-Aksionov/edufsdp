from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Shard

from gloofsdp.logging import logger
from gloofsdp.memory_tracker import memory_tracker

if TYPE_CHECKING:
    from gloofsdp.sharded_parameter import DTensorShardedParameter


class ForwardPreHook:
    """Pre-forward hook that gathers sharded parameters before computational pass."""

    def __init__(self, dtensor_param: DTensorShardedParameter):
        self.dtensor_param = dtensor_param

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        # Render transition from Sharded to Gathered
        logger.log_transition(
            event_name="all_gather",
            param_name=self.dtensor_param.name,
            total_elements=self.dtensor_param.global_shape.numel(),
            shard_range=self.dtensor_param.shard_range,
            before_gathered=False,
            after_gathered=True,
        )

        # 1. Gather full tensor
        full_tensor = self.dtensor_param.gather()

        # 2. CRITICAL: Register backward hook on the gathered tensor!
        # TODO (andrei aksionau): doesn't make sense to do it for every
        # forward call
        if full_tensor.requires_grad:
            full_tensor.register_hook(BackwardHook(self.dtensor_param))


class ForwardPostHook:
    """Post-forward hook that restores local parameter shards and frees full parameters."""

    def __init__(self, sharded_param: Any):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        # self.sharded_param.free()
        # logger.log_parameter_state(
        #     "HOOK: POST-FORWARD (Restored Local)",
        #     self.sharded_param.name,
        #     self.sharded_param.local_param.data,
        #     full_tensor=None,
        # )

        # Render transition from Gathered back to Sharded
        logger.log_transition(
            event_name="free",
            param_name=self.sharded_param.name,
            total_elements=self.sharded_param.global_shape.numel(),
            shard_range=self.sharded_param.shard_range,
            before_gathered=True,
            after_gathered=False,
        )
        self.sharded_param.free()


class BackwardHook:
    def __init__(self, dtensor_param: DTensorShardedParameter):
        self.dtensor_param = dtensor_param

    def __call__(self, grad: torch.Tensor) -> None:
        # Track full gradient allocation calculated by autograd
        memory_tracker.track_gradient(grad)

        # Render transition from Full Gradient to Reduced Local Gradient Shard
        logger.log_transition(
            event_name="reduce_scatter",
            param_name=f"grad({self.dtensor_param.name})",
            total_elements=self.dtensor_param.global_shape.numel(),
            shard_range=self.dtensor_param.shard_range,
            before_gathered=True,
            after_gathered=False,
        )
        # 1. Mark incoming gradient as Partial() because it needs reduction across ranks!
        grad_dtensor = DTensor.from_local(
            grad,
            device_mesh=self.dtensor_param.device_mesh,
            placements=[Partial()],  # <--- CRITICAL FIX!
            # placements=[Replicate()],  # <--- CRITICAL FIX!
        )
        # Changing [Replicate()] to [Partial()] in DTensor.from_local() will force DTensor to communicate gradients
        # across processes during backward, bringing Rank 0 and Rank 1 back into alignment
        # TODO (andrei aksionau): is it a thing only during backward or forward too?

        # Redistribute from Replicate() -> Shard(0)
        # DTensor automatically executes reduce_scatter & averages across the mesh!
        # TODO (andrei aksionau): ⬆ exactly, does it do it automatically or not?
        # Do I need to manually average gradients?
        # 2. Redistributing Partial() -> Shard(0) triggers a true REDUCE_SCATTER collective!
        sharded_grad_dtensor = grad_dtensor.redistribute(
            device_mesh=self.dtensor_param.device_mesh, placements=[Shard(dim=0)]
        )

        sharded_grad = sharded_grad_dtensor.to_local()

        # 3. Average the reduced sum across the mesh size
        # TODO (andrei aksionau): I thought that this should be done automatically.
        # Need to verify it. Basically the question is what the OP is used: Sum or Average
        world_size = self.dtensor_param.device_mesh.size()
        sharded_grad.div_(world_size)

        self.dtensor_param.local_param.grad = sharded_grad

        # Untrack full gradient and track local gradient shard
        memory_tracker.untrack_gradient(grad)
        memory_tracker.track_gradient(sharded_grad)

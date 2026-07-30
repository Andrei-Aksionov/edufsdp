from typing import Any

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from gloofsdp.collectives import reduce_scatter_tensor
from gloofsdp.logging import logger
from gloofsdp.memory_tracker import memory_tracker
from gloofsdp.sharded_parameter import DTensorShardedParameter, ShardedParameter


class ForwardPreHook:
    """Pre-forward hook that gathers sharded parameters before computational pass."""

    def __init__(self, sharded_param: Any):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        # full_param = self.sharded_param.gather()
        # logger.log_parameter_state(
        #     "HOOK: PRE-FORWARD (Gathered)",
        #     self.sharded_param.name,
        #     self.sharded_param.local_param.data,
        #     full_tensor=full_param.data,
        # )

        # Render transition from Sharded to Gathered
        logger.log_transition(
            event_name="all_gather",
            param_name=self.sharded_param.name,
            total_elements=self.sharded_param.global_shape.numel(),
            shard_range=self.sharded_param.shard_range,
            before_gathered=False,
            after_gathered=True,
        )
        self.sharded_param.gather()


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
    """Tensor backward hook that performs reduce-scatter on full gradients."""

    def __init__(self, sharded_param: Any):
        self.sharded_param = sharded_param

    def __call__(self, full_grad: torch.Tensor) -> None:
        # logger.log_gradient_state(
        #     "HOOK: BACKWARD (Before Reduce-Scatter)",
        #     self.sharded_param.name,
        #     full_grad=full_grad,
        #     local_grad=None,
        # )

        # local_grad_data = reduce_scatter_tensor(full_grad, self.sharded_param.shard_size)
        # self.sharded_param.local_param.grad = local_grad_data

        # logger.log_gradient_state(
        #     "HOOK: BACKWARD (After Reduce-Scatter)",
        #     self.sharded_param.name,
        #     full_grad=None,
        #     local_grad=local_grad_data,
        # )

        # Track full gradient allocation calculated by autograd
        memory_tracker.track_gradient(full_grad)

        # Render transition from Full Gradient to Reduced Local Gradient Shard
        logger.log_transition(
            event_name="reduce_scatter",
            param_name=f"grad({self.sharded_param.name})",
            total_elements=self.sharded_param.global_shape.numel(),
            shard_range=self.sharded_param.shard_range,
            before_gathered=True,
            after_gathered=False,
        )

        local_grad_data = reduce_scatter_tensor(full_grad, self.sharded_param.shard_size)
        self.sharded_param.local_param.grad = local_grad_data

        # Untrack full gradient and track local gradient shard
        memory_tracker.untrack_gradient(full_grad)
        memory_tracker.track_gradient(local_grad_data)


class BackwardHook1:
    def __init__(self, sharded_param: ShardedParameter):
        self.sharded_param = sharded_param

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()

        # 1. Prepare flat gradient buffer + padding
        flat_grad = grad.flatten()
        if self.sharded_param.padding_needed > 0:
            flat_grad = torch.cat([flat_grad, torch.zeros(self.sharded_param.padding_needed)])

        # 2. Allocate output tensor for this rank's reduced shard
        output_shard = torch.empty(flat_grad.numel() // world_size, dtype=grad.dtype)

        # 3. Explicit collective call: sum gradients across ranks into output_shard
        input_chunks = list(torch.chunk(flat_grad, world_size))
        dist.reduce_scatter(output_shard, input_chunks, op=dist.ReduceOp.SUM)

        # 4. Average gradient & assign to local parameter
        output_shard.div_(world_size)
        reshaped_shard = output_shard.view(self.sharded_param.local_param.shape)
        self.sharded_param.local_param.grad = reshaped_shard

        # Return None to avoid accumulating gradients on the temporary gathered parameter
        return None


class DTensorBackwardHook:
    def __init__(self, dtensor_param: DTensorShardedParameter):
        self.dtensor_param = dtensor_param

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        # Wrap the incoming full-size raw gradient as a Replicated DTensor
        grad_dtensor = DTensor.from_local(grad, device_mesh=self.dtensor_param.device_mesh, placements=[Replicate()])

        # Redistribute from Replicate() -> Shard(0)
        # DTensor automatically executes reduce_scatter & averages across the mesh!
        sharded_grad_dtensor = grad_dtensor.redistribute(
            device_mesh=self.dtensor_param.device_mesh, placements=[Shard(dim=0)]
        )

        # Extract the local tensor shard and attach to the local parameter
        self.dtensor_param.local_param.grad = sharded_grad_dtensor.to_local()

        return None

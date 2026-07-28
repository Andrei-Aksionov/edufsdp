from typing import Any

import torch
import torch.nn as nn

from gloofsdp.collectives import reduce_scatter_tensor
from gloofsdp.logging import logger


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

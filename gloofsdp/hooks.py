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
        full_param = self.sharded_param.gather()
        logger.log_parameter_state(
            "HOOK: PRE-FORWARD (Gathered)",
            self.sharded_param.name,
            self.sharded_param.local_param.data,
            full_tensor=full_param.data,
        )


class ForwardPostHook:
    """Post-forward hook that restores local parameter shards and frees full parameters."""

    def __init__(self, sharded_param: Any):
        self.sharded_param = sharded_param

    def __call__(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self.sharded_param.free()
        logger.log_parameter_state(
            "HOOK: POST-FORWARD (Restored Local)",
            self.sharded_param.name,
            self.sharded_param.local_param.data,
            full_tensor=None,
        )


class BackwardHook:
    """Tensor backward hook that performs reduce-scatter on full gradients."""

    def __init__(self, sharded_param: Any):
        self.sharded_param = sharded_param

    def __call__(self, full_grad: torch.Tensor) -> None:
        logger.log_gradient_state(
            "HOOK: BACKWARD (Before Reduce-Scatter)",
            self.sharded_param.name,
            full_grad=full_grad,
            local_grad=None,
        )

        local_grad_data = reduce_scatter_tensor(full_grad, self.sharded_param.shard_size)
        self.sharded_param.local_param.grad = local_grad_data

        logger.log_gradient_state(
            "HOOK: BACKWARD (After Reduce-Scatter)",
            self.sharded_param.name,
            full_grad=None,
            local_grad=local_grad_data,
        )

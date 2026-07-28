from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from gloofsdp.hooks import ForwardPostHook, ForwardPreHook
from gloofsdp.logging import logger
from gloofsdp.sharded_parameter import ShardedParameter


class EducationalFSDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.sharded_params: list[ShardedParameter] = []
        self._shard_parameters()
        self._register_hooks()

    def _shard_parameters(self) -> None:
        logger.log("Initializing parameter sharding across ranks...")

        for name, param in list(self.module.named_parameters()):
            sharded_param = ShardedParameter(
                name=name,
                module=self.module,
                param=param,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.sharded_params.append(sharded_param)

            # Initially place local parameter on module
            sharded_param.free()

    def _register_hooks(self) -> None:
        """Attaches explicit forward pre/post hooks for each sharded parameter."""
        for sp in self.sharded_params:
            # Helper to find target submodule for hook registration
            target_module = self._get_target_module(sp.name)
            target_module.register_forward_pre_hook(ForwardPreHook(sp))
            target_module.register_forward_hook(ForwardPostHook(sp))

    def _get_target_module(self, param_name: str) -> nn.Module:
        attrs = param_name.split(".")
        obj = self.module
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        return obj

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Hooks automatically handle gather before and free after execution!
        return self.module(*args, **kwargs)

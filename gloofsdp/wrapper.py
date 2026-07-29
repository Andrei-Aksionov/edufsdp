from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn

from gloofsdp.hooks import ForwardPostHook, ForwardPreHook
from gloofsdp.sharded_parameter import ShardedParameter
from gloofsdp.state_dict import StateDictHandler, StateDictType


class EducationalFSDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.sharded_params: Dict[str, ShardedParameter] = {}
        self._shard_parameters(self.module)
        self._register_hooks(self.module)

    def _shard_parameters(self, module: nn.Module, prefix: str = "") -> None:
        for name, param in list(module.named_parameters(recurse=False)):
            full_name = f"{prefix}{name}" if prefix else name
            sharded_param = ShardedParameter(
                name=full_name,
                module=module,
                param=param,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.sharded_params[full_name] = sharded_param

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            self._shard_parameters(child_module, prefix=child_prefix)

    def _register_hooks(self, module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            child_prefix = f"{prefix}{name}." if prefix else f"{name}."

            for p_name, param in child.named_parameters(recurse=False):
                full_param_name = f"{child_prefix}{p_name}"
                if full_param_name in self.sharded_params:
                    sharded = self.sharded_params[full_param_name]
                    child.register_forward_pre_hook(ForwardPreHook(sharded))
                    child.register_forward_hook(ForwardPostHook(sharded))

            self._register_hooks(child, prefix=child_prefix)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def state_dict(
        self,
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> Dict[str, torch.Tensor]:
        return StateDictHandler.get_state_dict(self.sharded_params, state_dict_type)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> None:
        StateDictHandler.load_state_dict(self.sharded_params, state_dict, state_dict_type)

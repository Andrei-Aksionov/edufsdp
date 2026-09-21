"""
Educational FSDP Wrapper module.

Transforms target `nn.Module` by sharding parameters across ranks and attaching
pre/post execution hooks to handle automatic parameter gathering and freeing.
"""

from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from gloofsdp.hooks import BackwardPostHook, BackwardPreHook, ForwardPostHook, ForwardPreHook
from gloofsdp.sharded_parameter import DTensorShardedParameter
from gloofsdp.state_dict import StateDictHandler, StateDictType


class EducationalFSDP(nn.Module):
    """
    Educational Fully Sharded Data Parallel wrapper.
    """

    def __init__(self, module: nn.Module, device_mesh: DeviceMesh | None = None) -> None:
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device_mesh = device_mesh

        self.sharded_params: Dict[str, DTensorShardedParameter] = {}

        # 1. Transform parameters into DTensorShardedParameters
        self._shard_parameters(self.module)

        # 2. Attach module execution hooks for pre/post forward lifecycle
        self._register_hooks(self.module)

    def _shard_parameters(self, module: nn.Module, prefix: str = "") -> None:
        """
        Traverses the module tree to replace unsharded torch.nn.Parameters with DTensorShardedParameters.

        Order: Processes direct parameters first, then recurses into child submodules.
        Target: Operates on individual torch.nn.Parameters.
        """
        # 1. Outer Loop: Process parameters directly attached to this `module` instance.
        # We use recurse=False to prevent grabbing parameters belonging to submodules (e.g. child Linear layers).
        for name, param in list(module.named_parameters(recurse=False)):
            full_name = f"{prefix}{name}" if prefix else name

            # Convert standard parameter into a sharded parameter (splits tensor across ranks)
            sharded_param = DTensorShardedParameter(
                name=full_name,
                module=module,
                param=param,
                device_mesh=self.device_mesh,
            )

            # Track in FSDP state
            self.sharded_params[full_name] = sharded_param

        # 2. Inner / Sequential Loop: Recurse into child submodules to shard their parameters.
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            self._shard_parameters(child_module, prefix=child_prefix)

    def _register_hooks(self, module: nn.Module, prefix: str = "") -> None:
        """
        Traverses the module tree to attach forward execution hooks to submodules.

        Order: Processes child submodules first to attach hooks to nn.Module containers.
        Target: Operates on nn.Module containers (PyTorch hooks must attach to modules, not parameters).
        """
        # 1. Outer Loop: Identify target child modules (containers) that execute forward passes.
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."

            # 2. Inner Loop: Find all sharded parameters owned specifically by this `child_module`.
            child_params = [
                self.sharded_params[f"{child_prefix}{p_name}"]
                for p_name, _ in child_module.named_parameters(recurse=False)
                if f"{child_prefix}{p_name}" in self.sharded_params
            ]

            # 3. Attach execution hooks onto `child_module` for every parameter it requires.
            # Before child_module.forward(): All-Gather sharded parameters into full parameters.
            # After child_module.forward(): Free full parameters to release peak GPU memory.
            for sharded in child_params:
                child_module.register_forward_pre_hook(ForwardPreHook(sharded))
                child_module.register_forward_hook(ForwardPostHook(sharded))
                child_module.register_full_backward_pre_hook(BackwardPreHook(sharded))
                child_module.register_full_backward_hook(BackwardPostHook(sharded))

            # 4. Recurse down to register hooks on submodules within `child_module`.
            self._register_hooks(child_module, prefix=child_prefix)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegates forward pass execution to wrapped inner module."""
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

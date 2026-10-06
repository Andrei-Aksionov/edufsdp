"""
Educational FSDP Wrapper module.

Transforms a target `nn.Module` by recursively sharding parameters across ranks
and attaching pre/post execution hooks for automatic parameter gathering and freeing.

Execution lifecycle for each submodule:

            Forward Pass                                              Backward Pass
 ┌──────────────────────────────────────┐                  ┌──────────────────────────────────────┐
 │ ForwardPreHook                       │                  │ BackwardPreHook                      │
 │   --> all_gather()                   │                  │   --> all_gather()                   │
 │   Gathers weight & bias to full size │                  │   Gathers weight & bias to full size │
 │   [Full memory]                      │                  │   [Full memory]                      │
 └──────────────────┬───────────────────┘                  └──────────────────┬───────────────────┘
                    │                                                         │
                    ▼                                                         ▼
 ┌──────────────────────────────────────┐                  ┌──────────────────────────────────────┐
 │ Forward Compute                      │                  │ Backward Compute                     │
 │   Executes layer ops (e.g. F.linear) │                  │   Calculates weight, bias & input    │
 │   using unsharded full parameters    │                  │   gradients using full parameters    │
 └──────────────────┬───────────────────┘                  └──────────────────┬───────────────────┘
                    │                                                         │
                    ▼                                                         ▼
 ┌──────────────────────────────────────┐                  ┌──────────────────────────────────────┐
 │ ForwardPostHook                      │                  │ reduce_scatter()                     │
 │   --> free()                         │                  │   Averages & shards weight & bias    │
 │   Drops full weight & bias tensors,  │                  │   gradients across ranks into local  │
 │   restoring DTensors [1/N memory]    │                  │   1/N DTensor shard buffers          │
 └──────────────────────────────────────┘                  └──────────────────┬───────────────────┘
                                                                              │
                                                                              ▼
                                                           ┌──────────────────────────────────────┐
                                                           │ BackwardPostHook                     │
                                                           │   --> free()                         │
                                                           │   Drops full weight & bias tensors,  │
                                                           │   restoring DTensors [1/N memory]    │
                                                           └──────────────────────────────────────┘
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from gloofsdp.hooks import (
    BackwardPostHook,
    BackwardPreHook,
    ForwardPostHook,
    ForwardPreHook,
)
from gloofsdp.sharded_parameter import ShardedParameter
from gloofsdp.state_dict import StateDictManager, StateDictType


class FSDP(nn.Module):
    """
    Fully Sharded Data Parallel wrapper designed for clarity and learning.

    Wraps a PyTorch `nn.Module`, sharding its parameters across a `DeviceMesh`
    and attaching custom autograd hooks to manage collective operations.

    Args:
        module (nn.Module): Target PyTorch module to shard and wrap.
        device_mesh (DeviceMesh): PyTorch DeviceMesh defining process rank layout.
    """

    def __init__(self, module: nn.Module, device_mesh: DeviceMesh) -> None:
        super().__init__()
        self.module = module
        self.device_mesh = device_mesh

        # Single-pass transformation: shards parameters & registers hooks in one go
        self._wrap_module_tree(self.module)

    def _wrap_module_tree(self, module: nn.Module, prefix: str = "") -> None:
        """
        Traverses the module tree recursively.

        For each submodule:
          1. Recursively traverses child modules (bottom-up traversal).
          2. Shards direct parameters into `ShardedParameter` wrappers.
          3. Attaches forward and backward execution hooks directly to the submodule.

        Args:
            module (nn.Module): Current module instance in the traversal.
            prefix (str): Accumulated dot-separated FQN path (e.g., 'layer1.').
        """
        # 1. Post-order recursion: process child submodules first
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            self._wrap_module_tree(child_module, prefix=child_prefix)

        # Dictionary attached to submodule storing local parameter wrappers
        module._fsdp_sharded_params = {}

        # 2. Process parameters owned directly by this module instance
        for param_name, param in list(module.named_parameters(recurse=False)):
            full_name = f"{prefix}{param_name}" if prefix else param_name

            sharded_param = ShardedParameter(
                fqn=full_name,
                module=module,
                param=param,
                device_mesh=self.device_mesh,
            )

            module._fsdp_sharded_params[param_name] = sharded_param

            # 3. Attach forward & backward hooks to trigger all_gather and free
            module.register_forward_pre_hook(ForwardPreHook(sharded_param))
            module.register_forward_hook(ForwardPostHook(sharded_param))
            module.register_full_backward_pre_hook(BackwardPreHook(sharded_param))
            module.register_full_backward_hook(BackwardPostHook(sharded_param))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegates forward pass execution to the inner wrapped module."""
        return self.module(*args, **kwargs)

    def state_dict(
        self,
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> Dict[str, torch.Tensor]:
        """
        Exports module state dict in either `LOCAL` (sharded) or `FULL` (unsharded) format.

        Args:
            state_dict_type (StateDictType): Target format (StateDictType.LOCAL or StateDictType.FULL).

        Returns:
            Dict[str, torch.Tensor]: Map of parameter FQNs to tensor shards or gathered full tensors.
        """
        return StateDictManager.get_state_dict(self.module, state_dict_type)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> Any:
        """
        Loads state dictionary checkpoint into sharded parameters.

        Args:
            state_dict (Dict[str, torch.Tensor]): Checkpoint mapping parameter names to tensors.
            state_dict_type (StateDictType): Format of incoming state_dict (`LOCAL` or `FULL`).
        """
        StateDictManager.load_state_dict(self.module, state_dict, state_dict_type)

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
    """Fully Sharded Data Parallel wrapper designed for clarity and learning."""

    def __init__(self, module: nn.Module, device_mesh: DeviceMesh) -> None:
        """
        Wraps a PyTorch `nn.Module`.

        Shards its parameters across a `DeviceMesh` and attaches custom autograd
        hooks to manage collective operations.

        Args:
            module: Target PyTorch module to shard and wrap.
            device_mesh: PyTorch DeviceMesh defining process rank layout.

        """
        super().__init__()
        self.inner_module = module
        self.device_mesh = device_mesh

        # Single-pass transformation: shards parameters & registers hooks in one go
        self._wrap_module_tree(self.inner_module)

    def _wrap_module_tree(self, module: nn.Module, prefix: str = "") -> None:
        """
        Traverses the module tree recursively.

        For each submodule:
          1. Recursively traverses child modules (bottom-up traversal).
          2. Shards direct parameters into `ShardedParameter` wrappers.
          3. Attaches forward and backward execution hooks directly to the submodule.

        Args:
            module: Current module instance in the traversal.
            prefix: Accumulated dot-separated FQN path (e.g., 'layer1.').

        """
        # 1. Post-order recursion: process child submodules first
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            self._wrap_module_tree(child_module, prefix=child_prefix)

        # Dictionary attached to submodule storing local parameter wrappers
        module._fsdp_sharded_parameters = {}

        # 2. Process parameters owned directly by this module instance
        for parameter_name, parameter in list(module.named_parameters(recurse=False)):
            full_name = f"{prefix}{parameter_name}" if prefix else parameter_name

            sharded_parameter = ShardedParameter(
                fqn=full_name,
                module=module,
                parameter=parameter,
                device_mesh=self.device_mesh,
            )

            module._fsdp_sharded_parameters[parameter_name] = sharded_parameter

            # 3. Attach forward & backward hooks to trigger all_gather and free
            module.register_forward_pre_hook(ForwardPreHook(sharded_parameter))
            module.register_forward_hook(ForwardPostHook(sharded_parameter))
            module.register_full_backward_pre_hook(BackwardPreHook(sharded_parameter))
            module.register_full_backward_hook(BackwardPostHook(sharded_parameter))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Delegates forward pass execution to the inner wrapped module.

        Args:
            *args: Positional inputs passed to the inner module.
            **kwargs: Keyword inputs passed to the inner module.

        Returns:
            Output computation graph produced by the inner module forward pass.

        """
        return self.inner_module(*args, **kwargs)

    @property
    def sharded_parameters(self) -> dict[str, ShardedParameter]:
        """Dynamically collects all ShardedParameter objects across submodules by searching through module attributes."""
        sharded_map: dict[str, ShardedParameter] = {}

        for module_name, submodule in self.inner_module.named_modules():
            # Check if this submodule holds any registered sharded parameter objects
            if hasattr(submodule, "_fsdp_sharded_parameters"):
                for parameter_name, parameter_wrapper in submodule._fsdp_sharded_parameters.items():
                    # Construct fully qualified parameter name (e.g., 'layer1.weight')
                    full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                    sharded_map[full_name] = parameter_wrapper

        return sharded_map

    def state_dict(
        self,
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> Dict[str, torch.Tensor]:
        """
        Exports module state dict in either `LOCAL` (sharded) or `FULL` (unsharded) format.

        Args:
            state_dict_type: Target format (StateDictType.LOCAL or StateDictType.FULL).

        Returns:
            Map of parameter FQNs to tensor shards or gathered full tensors.

        """
        return StateDictManager.get_state_dict(self.inner_module, self.sharded_parameters, state_dict_type)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> Any:
        """
        Loads state dictionary checkpoint into sharded parameters.

        Args:
            state_dict: Checkpoint mapping parameter names to tensors.
            state_dict_type: Format of incoming state_dict (`LOCAL` or `FULL`).

        """
        StateDictManager.load_state_dict(self.inner_module, self.sharded_parameters, state_dict, state_dict_type)

"""
Educational FSDP Wrapper module.

Transforms a target `nn.Module` by recursively sharding parameters across ranks
and attaching pre/post execution hooks for automatic parameter gathering and freeing.

Execution lifecycle for each submodule:

            Forward Pass                                          Backward Pass
 ┌──────────────────────────────────────┐              ┌──────────────────────────────────────┐
 │ ForwardPreHook                       │              │ BackwardPreHook                      │
 │   --> all_gather()                   │              │   --> all_gather()                   │
 │   Gathers weight & bias to full size │              │   Gathers weight & bias to full size │
 │   [Full memory]                      │              │   [Full memory]                      │
 └──────────────────┬───────────────────┘              └──────────────────┬───────────────────┘
                    │                                                     │
                    ▼                                                     ▼
 ┌──────────────────────────────────────┐              ┌──────────────────────────────────────┐
 │ Forward Compute                      │              │ Backward Compute                     │
 │   Executes layer ops (e.g. F.linear) │              │   Calculates weight, bias & input    │
 │   using unsharded full parameters    │              │   gradients using full parameters    │
 └──────────────────┬───────────────────┘              └──────────────────┬───────────────────┘
                    │                                                     │
                    ▼                                                     ▼
 ┌──────────────────────────────────────┐              ┌──────────────────────────────────────┐
 │ ForwardPostHook                      │              │ reduce_scatter()                     │
 │   --> free()                         │              │   Averages & shards weight & bias    │
 │   Drops full weight & bias tensors,  │              │   gradients across ranks into local  │
 │   restoring DTensors [1/N memory]    │              │   1/N DTensor shard buffers          │
 └──────────────────────────────────────┘              └──────────────────┬───────────────────┘
                                                                          │
                                                                          ▼
                                                       ┌──────────────────────────────────────┐
                                                       │ BackwardPostHook                     │
                                                       │   --> free()                         │
                                                       │   Drops full weight & bias tensors,  │
                                                       │   restoring DTensors [1/N memory]    │
                                                       └──────────────────────────────────────┘
"""

from typing import TYPE_CHECKING, Any

import torch.nn as nn
from torch import Tensor

from gloofsdp.hooks import (
    BackwardPostHook,
    BackwardPreHook,
    ForwardPostHook,
    ForwardPreHook,
)
from gloofsdp.sharded_parameter import ShardedParameter
from gloofsdp.state_dict import StateDictManager, StateDictType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.distributed.device_mesh import DeviceMesh


class FSDP[ModuleT: nn.Module](nn.Module):
    """
    Fully Sharded Data Parallel wrapper over a PyTorch `nn.Module`.

    Shards module parameters across a process group `DeviceMesh` using a
    centralized registry and manages collective operations (`all_gather`,
    `reduce_scatter`) via autograd hooks.

    Attributes:
        inner_module (ModuleT): Wrapped target module instance.
        device_mesh (DeviceMesh): PyTorch DeviceMesh defining process rank topology.
        sharded_parameters (dict[str, ShardedParameter]): Centralized registry
            mapping parameter FQNs to their corresponding passive wrappers.

    """

    def __init__(self, module: ModuleT, device_mesh: DeviceMesh) -> None:
        """
        Initializes FSDP wrapper and shards child parameters.

        Args:
            module: Target PyTorch module to shard and wrap.
            device_mesh: PyTorch DeviceMesh defining process rank layout.

        """
        super().__init__()
        self.inner_module: ModuleT = module
        self.device_mesh = device_mesh

        self.sharded_parameters: dict[str, ShardedParameter] = {}
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

        # 2. Process parameters owned directly by this module instance
        for parameter_name, parameter in list(module.named_parameters(recurse=False)):
            fqn = f"{prefix}{parameter_name}" if prefix else parameter_name

            sharded_parameter = ShardedParameter(
                fqn=fqn,
                module=module,
                parameter=parameter,
                device_mesh=self.device_mesh,
            )
            self.sharded_parameters[fqn] = sharded_parameter

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

    def state_dict(
        self,
        *args: Any,
        state_dict_type: StateDictType = StateDictType.LOCAL,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """
        Exports module state dict in either `LOCAL` (sharded) or `FULL` (unsharded) format.

        Args:
            *args: Positional arguments matching `nn.Module.state_dict`.
            state_dict_type: Target format (`StateDictType.LOCAL` or `StateDictType.FULL`).
            **kwargs: Keyword arguments matching `nn.Module.state_dict` (prefix, keep_vars).

        Returns:
            Map of parameter FQNs to tensor shards or gathered full tensors.

        """
        return StateDictManager.get_state_dict(
            self.inner_module, self.sharded_parameters, state_dict_type
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> None:
        """
        Loads state dictionary checkpoint into sharded parameters.

        Args:
            state_dict: Checkpoint mapping parameter names to tensors.
            strict: Whether to strictly enforce key matching in state_dict.
            assign: Whether to assign tensors directly rather than copying in-place.
            state_dict_type: Format of incoming state_dict (`LOCAL` or `FULL`).

        """
        return StateDictManager.load_state_dict(
            self.inner_module, self.sharded_parameters, state_dict, state_dict_type
        )

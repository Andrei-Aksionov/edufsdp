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


class EducationalFSDP1(nn.Module):
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


class EducationalFSDP(nn.Module):
    def __init__(self, module: nn.Module, device_mesh: DeviceMesh | None = None) -> None:
        super().__init__()
        self.module = module
        self.device_mesh = device_mesh

        # Single-pass transformation: shards parameters & registers hooks in one go
        self._wrap_module_tree(self.module)

    def _wrap_module_tree(self, module: nn.Module, prefix: str = "") -> None:
        """
        Traverses the module tree once. For each leaf module:
        1. Shards its direct parameters into DTensorShardedParameters.
        2. Attaches forward and backward execution hooks directly to the module.
        """
        # 1. First recurse to child submodules
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}{child_name}." if prefix else f"{child_name}."
            self._wrap_module_tree(child_module, prefix=child_prefix)

        module._fsdp_sharded_params = {}

        # 2. Process parameters owned directly by this module instance
        for param_name, param in list(module.named_parameters(recurse=False)):
            full_name = f"{prefix}{param_name}" if prefix else param_name

            # Instantiate sharded parameter management wrapper
            sharded_param = DTensorShardedParameter(
                name=full_name,
                module=module,
                param=param,
                device_mesh=self.device_mesh,
            )

            # Save wrapper reference locally on the submodule
            module._fsdp_sharded_params[param_name] = sharded_param

            # 2.1. Attach hooks immediately to this module using local parameter references
            module.register_forward_pre_hook(ForwardPreHook(sharded_param))
            module.register_forward_hook(ForwardPostHook(sharded_param))
            module.register_full_backward_pre_hook(BackwardPreHook(sharded_param))
            module.register_full_backward_hook(BackwardPostHook(sharded_param))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegates forward pass execution to wrapped inner module."""
        return self.module(*args, **kwargs)

    def _get_sharded_param_map(self) -> Dict[str, DTensorShardedParameter]:
        """
        Dynamically collects all DTensorShardedParameter objects across
        submodules by searching through module attributes.
        """
        sharded_map: Dict[str, DTensorShardedParameter] = {}

        for mod_name, submodule in self.module.named_modules():
            # Check if this submodule holds any registered sharded parameter objects
            if hasattr(submodule, "_fsdp_sharded_params"):
                for p_name, param_wrapper in submodule._fsdp_sharded_params.items():
                    # Construct fully qualified parameter name (e.g., 'layer1.weight')
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    sharded_map[full_name] = param_wrapper

        return sharded_map

    def state_dict(
        self,
        state_dict_type: StateDictType = StateDictType.LOCAL,
    ) -> Dict[str, torch.Tensor]:
        """
        Exports the module state dictionary in LOCAL (sharded) or FULL (unsharded) format.
        """
        sharded_params = self._get_sharded_param_map()
        return StateDictHandler.get_state_dict(sharded_params, state_dict_type)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        state_dict_type: StateDictType = StateDictType.LOCAL,
        strict: bool = True,
    ) -> Any:
        """
        Loads a state dictionary in LOCAL or FULL format into sharded parameter buffers.
        """
        sharded_params = self._get_sharded_param_map()
        StateDictHandler.load_state_dict(sharded_params, state_dict, state_dict_type)

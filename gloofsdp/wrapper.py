from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from gloofsdp.collectives import all_gather_parameter
from gloofsdp.logging import logger


def _get_nested_attr(obj: object, attr_path: str) -> None:
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _set_nested_attr(obj: object, attr_path: str, value: object) -> None:
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


def _del_nested_attr(obj: object, attr_path: str) -> None:
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    delattr(obj, attrs[-1])


class EducationalFSDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Metadata storage for reconstruction
        self._param_metadata = {}

        self._shard_parameters()

    def _shard_parameters(self):
        """
        Flattens, pads, and shards every parameter within the module.
        """
        logger.log("Initializing parameter sharding across ranks...")

        for name, param in list(self.module.named_parameters()):
            # Save original metadata
            orig_shape = param.shape
            orig_data = param.data

            # Flatten and handle padding if not divisible by world_size
            flat_param = orig_data.flatten()
            padding_needed = (self.world_size - (flat_param.numel() % self.world_size)) % self.world_size
            if padding_needed > 0:
                flat_param = torch.cat([flat_param, torch.zeros(padding_needed, device=flat_param.device)])

            # Divide into chunks chunks
            chunks = torch.chunk(flat_param, self.world_size)
            local_shard_data = chunks[self.rank].clone()

            # Save metadata required for the gather stage
            self._param_metadata[name] = {
                "orig_shape": orig_shape,
                "local_shard": local_shard_data,
                "padding_needed": padding_needed,
            }

            # Replace the parameter with our local shard to minimize initial storage footprint
            # For Milestone 1, we register it as a normal parameter so optimizers can track it.
            _del_nested_attr(self.module, name)
            _set_nested_attr(self.module, name, nn.Parameter(local_shard_data, requires_grad=param.requires_grad))

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # 1. Log states BEFORE gather
        for name, meta in self._param_metadata.items():
            current_local_tensor = _get_nested_attr(self.module, name)
            logger.log_parameter_state("BEFORE FORWARD (Sharded)", name, current_local_tensor, full_tensor=None)

        # 2. GATHER: Reconstruct full weights for computational phase
        gathered_tensors = {}
        for name, meta in self._param_metadata.items():
            current_local_tensor = _get_nested_attr(self.module, name)

            # Trigger our explicit collective wrapper
            full_tensor = all_gather_parameter(current_local_tensor.data, meta["orig_shape"])
            gathered_tensors[name] = full_tensor

            # Temporarily attach the full tensor to the module for the forward execution pass
            _del_nested_attr(self.module, name)
            _set_nested_attr(
                self.module, name, nn.Parameter(full_tensor, requires_grad=current_local_tensor.requires_grad)
            )

            # Log states DURING computation
            logger.log_parameter_state("DURING FORWARD (Gathered)", name, current_local_tensor, full_tensor=full_tensor)

        # 3. Compute execution forward pass
        output = self.module(*args, **kwargs)

        # 4. FREE: Discard gathered full parameters, reverting tracking back to the local shard
        for name, meta in self._param_metadata.items():
            # Extract local shard from our structural tracking metadata store
            local_shard_data = meta["local_shard"]

            _del_nested_attr(self.module, name)
            _set_nested_attr(self.module, name, nn.Parameter(local_shard_data, requires_grad=True))

            # Log states AFTER freeing
            logger.log_parameter_state("AFTER FORWARD (Freed)", name, local_shard_data, full_tensor=None)

        return output

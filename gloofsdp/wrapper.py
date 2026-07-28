from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

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

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # 1. Log BEFORE gather
        for sp in self.sharded_params:
            logger.log_parameter_state("BEFORE FORWARD (Sharded)", sp.name, sp.local_param.data, full_tensor=None)

        # 2. GATHER: Gather full tensors onto module
        for sp in self.sharded_params:
            full_param = sp.gather()
            logger.log_parameter_state(
                "DURING FORWARD (Gathered)", sp.name, sp.local_param.data, full_tensor=full_param.data
            )

        # 3. Compute forward pass
        output = self.module(*args, **kwargs)

        # 4. FREE: Restore local_param on module (Fixes loop iteration updates!)
        for sp in self.sharded_params:
            sp.free()
            logger.log_parameter_state("AFTER FORWARD (Restored Local)", sp.name, sp.local_param.data, full_tensor=None)

        return output

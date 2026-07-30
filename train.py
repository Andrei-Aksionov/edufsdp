import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from gloofsdp.logging import logger
from gloofsdp.memory_tracker import memory_tracker
from gloofsdp.state_dict import StateDictType
from gloofsdp.wrapper import EducationalFSDP
from model import MLP


def init_distributed() -> None:
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    dist.init_process_group(backend="gloo")
    torch.manual_seed(42 + dist.get_rank())


def main() -> None:
    init_distributed()
    mesh = init_device_mesh("cpu", mesh_shape=(dist.get_world_size(),))

    raw_model = MLP()
    fsdp_model = EducationalFSDP(raw_model, device_mesh=mesh)

    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    inputs = torch.randn(2, 8)
    targets = torch.randn(2, 4)

    logger.log_ordered("=== MEMORY SUMMARY BEFORE FORWARD ===\n" + memory_tracker.get_summary())

    outputs = fsdp_model(inputs)
    loss = criterion(outputs, targets)

    logger.log_ordered("=== MEMORY SUMMARY AFTER FORWARD ===\n" + memory_tracker.get_summary())

    optimizer.zero_grad()
    loss.backward()

    logger.log_ordered("=== MEMORY SUMMARY AFTER BACKWARD ===\n" + memory_tracker.get_summary())

    optimizer.step()

    check_sd_loading(fsdp_model, mesh)

    dist.destroy_process_group()


def check_sd_loading(fsdp_model: nn.Module, device_mesh: DeviceMesh) -> None:

    # 1. Test Local State Dict
    local_sd = fsdp_model.state_dict(state_dict_type=StateDictType.LOCAL)
    logger.log_ordered("=== LOCAL STATE DICT ===\n" + "\n".join([f"{k}: shape {v.shape}" for k, v in local_sd.items()]))

    # 2. Test Full State Dict
    full_sd = fsdp_model.state_dict(state_dict_type=StateDictType.FULL)
    if dist.get_rank() == 0:
        logger.log(
            "=== FULL STATE DICT (Rank 0) ===\n" + "\n".join([f"{k}: shape {v.shape}" for k, v in full_sd.items()])
        )

    # Verify reloading full state dict onto a fresh model instance
    fresh_fsdp_model = EducationalFSDP(MLP(), device_mesh=device_mesh)
    fresh_fsdp_model.load_state_dict(full_sd, state_dict_type=StateDictType.FULL)
    logger.log_ordered("Successfully loaded Full State Dict into fresh model!")


if __name__ == "__main__":
    main()

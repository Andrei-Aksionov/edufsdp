import os
import platform

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from gloofsdp.logging import logger
from gloofsdp.model import MLP
from gloofsdp.state_dict import StateDictType
from gloofsdp.wrapper import EducationalFSDP


def init_distributed() -> None:
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    current_os = platform.system()
    if current_os == "Darwin":  # macOS
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    elif current_os == "Linux":  # Linux / Lightning AI Studio
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    dist.init_process_group(backend="gloo")
    torch.manual_seed(42 + dist.get_rank())


def main() -> None:
    init_distributed()
    mesh = init_device_mesh("cpu", mesh_shape=(dist.get_world_size(),))

    raw_model = MLP()
    fsdp_model = EducationalFSDP(raw_model, device_mesh=mesh)

    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    inputs = torch.randn(2, 8, requires_grad=True)
    targets = torch.randn(2, 4)

    for _ in range(1):
        logger.log_rank0(f"\n\n {'=' * 55} Forward pass {'=' * 55}\n")

        outputs = fsdp_model(inputs)
        loss = criterion(outputs, targets)
        logger.log_rank0("\n\n######## LOSS VALUES: ########\n")
        logger.log_ordered(f"Loss value: {loss.item():.4f}")
        logger.log_rank0(f"\n{'#' * 30}")

        optimizer.zero_grad()
        logger.log_rank0(f"\n\n {'=' * 55} Backward pass {'=' * 55}\n")
        logger.reset_step()
        loss.backward()

        optimizer.step()

    # check_sd_loading(fsdp_model, mesh)

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

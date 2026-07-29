import os

import torch
import torch.distributed as dist
import torch.nn as nn

from gloofsdp.logging import logger
from gloofsdp.memory_tracker import memory_tracker
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

    raw_model = MLP()
    fsdp_model = EducationalFSDP(raw_model)

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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

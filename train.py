import os

import torch
import torch.distributed as dist
import torch.nn as nn

from gloofsdp.logging import logger
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

    logger.log("Spawning standard architecture model...")
    raw_model = MLP()

    # Wrap model with GlooFSDP
    fsdp_model = EducationalFSDP(raw_model)

    # The optimizer ONLY sees local parameter shards!
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    inputs = torch.randn(2, 8)
    targets = torch.randn(2, 4)

    for _ in range(3):
        # Step 1: Forward
        logger.log("=== STARTING FORWARD PASS ===")
        outputs = fsdp_model(inputs)
        loss = criterion(outputs, targets)
        logger.log(f"Forward completed. Loss: {loss.item():.4f}")

        # Step 2: Backward
        logger.log("=== STARTING BACKWARD PASS ===")
        optimizer.zero_grad()
        loss.backward()
        logger.log("Backward completed.")

        # Step 3: Optimizer Step
        logger.log("=== EXECUTING OPTIMIZER STEP ON LOCAL SHARDS ===")
        optimizer.step()
        logger.log("Optimizer step completed successfully!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

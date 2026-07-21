import os

import torch
import torch.distributed as dist
import torch.nn as nn

from gloofsdp.logging import logger
from gloofsdp.wrapper import EducationalFSDP
from model import MLP


def init_distributed() -> None:
    # Fallback default values for local development loops without torchrun script calls
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(backend="gloo")
    torch.manual_seed(42 + dist.get_rank())


def main() -> None:
    init_distributed()

    logger.log("Spawning standard architecture model...")
    raw_model = MLP()

    # Wrap using our custom simulated execution layer
    fsdp_model = EducationalFSDP(raw_model)

    # Local shard optimizer setup
    criterion = nn.MSELoss()

    # Synthesize consistent batch samples
    inputs = torch.randn(2, 8)
    targets = torch.randn(2, 4)

    logger.log("Executing Step 1 Forward Pass...")
    outputs = fsdp_model(inputs)

    # Milestone 1: Compute loss
    loss = criterion(outputs, targets)
    logger.log(f"Forward completed successfully. Step Loss Value: {loss.item():.4f}")

    # Clean termination
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

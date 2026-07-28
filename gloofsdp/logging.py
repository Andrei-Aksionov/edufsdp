import os

import torch


class RankLogger:
    def __init__(self):
        # torchrun sets these env vars automatically
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    def log(self, msg: str) -> None:
        print(self.format_log(msg), flush=True)

    def format_log(self, msg: str) -> str:
        return f"[Rank {self.rank}/{self.world_size}] {msg}"

    def log_parameter_state(
        self, stage: str, param_name: str, local_tensor: torch.Tensor, full_tensor: torch.Tensor | None = None
    ) -> None:
        local_elements = local_tensor.numel() if local_tensor is not None else 0
        full_elements = full_tensor.numel() if full_tensor is not None else 0

        status = f"Status: {'GATHERED' if full_tensor is not None else 'SHARDED'}"

        msg = (
            f"--- {stage} | Parameter: {param_name} ---\n"
            f"    -> Local Shard Shape: {list(local_tensor.shape)} ({local_elements} elements)\n"
            f"    -> Active Tensor Shape: {list(full_tensor.shape if full_tensor is not None else local_tensor.shape)}\n"
            f"    -> {status} (Total simulated global elements: {full_elements if full_elements else local_elements * self.world_size})"
        )
        self.log(msg)

    def log_gradient_state(
        self, stage: str, param_name: str, full_grad: torch.Tensor | None, local_grad: torch.Tensor | None
    ) -> None:
        full_elements: int = full_grad.numel() if full_grad is not None else 0
        local_elements: int = local_grad.numel() if local_grad is not None else 0

        msg: str = (
            f"--- {stage} | Gradient: {param_name} ---\n"
            f"    -> Full Gradient Elements: {full_elements}\n"
            f"    -> Local Gradient Shard Elements: {local_elements}\n"
            f"    -> Status: {'REDUCE_SCATTERED TO LOCAL SHARD' if local_grad is not None else 'FULL GRADIENT COMPUTED'}"
        )
        self.log(msg)


logger = RankLogger()

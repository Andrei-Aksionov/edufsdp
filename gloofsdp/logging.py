import os

import torch.distributed as dist

from gloofsdp.visualizer import ASCIIVisualizer


class RankLogger:
    def __init__(self):
        # torchrun sets these env vars automatically
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.last_layer_prefix: str | None = None

        self.log_hooks = False

    def log(self, msg: str) -> None:
        # TODO: why do we need to flush?
        print(self.format_log(msg), flush=False)

    def log_rank0(self, msg: str) -> None:
        if self.rank == 0:
            print(msg, flush=False)

    def format_log(self, msg: str) -> str:
        return f"[Rank {self.rank}/{self.world_size}] {msg}"

    def log_ordered(self, msg: str, format_log: bool = True) -> None:
        """
        Guarantees sequentially ordered stdout across all ranks by passing
        a token through process barriers.
        """
        if not dist.is_initialized():
            if msg.strip():
                self.log(msg)
            return

        for rank in range(self.world_size):
            if self.rank == rank and msg.strip():
                formatted = "\n".join(self.format_log(line) if format_log else line for line in msg.splitlines())
                print(formatted, flush=False)

            dist.barrier()

    def _extract_layer_prefix(self, param_name: str) -> str:
        """
        Extracts top-level or submodule layer prefix from parameter name.
        Example: 'layer1.weight' -> 'layer1', 'encoder.layer.0.bias' -> 'encoder.layer.0'
        """
        parts = param_name.split(".")
        return parts[0] if len(parts) > 1 else param_name

    def reset_step(self) -> None:
        """Call at the start or end of a training step to reset layer boundary tracking."""
        self.last_layer_prefix = None

    def log_transition(
        self,
        event_name: str,
        param_name: str,
        total_elements: int,
        shard_range: tuple[int, int],
        before_gathered: bool,
        after_gathered: bool,
        shape: str | None = None,
    ) -> None:
        if not self.log_hooks:
            return

        before_bar = ASCIIVisualizer.render_shard_bar(
            total_elements=total_elements,
            active_range=shard_range,
            is_gathered=before_gathered,
        )
        after_bar = ASCIIVisualizer.render_shard_bar(
            total_elements=total_elements,
            active_range=shard_range,
            is_gathered=after_gathered,
        )

        current_prefix = self._extract_layer_prefix(param_name)

        if self.last_layer_prefix is None or current_prefix != self.last_layer_prefix:
            self.log_rank0(f"\n {' ' * 51} {'-' * 5} {current_prefix} {'-' * 5}\n")

        self.last_layer_prefix = current_prefix

        msg = (
            f"{'-' * 125}\n"
            f"Rank {self.rank} | {param_name:<20} | {event_name:<15} | "
            f"State: {before_bar} -> {after_bar}"
            f" | Shape: {shape}\n"
        )
        self.log_ordered(msg, format_log=False)


logger = RankLogger()

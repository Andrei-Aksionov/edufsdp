import os

import torch
import torch.distributed as dist


class ASCIIVisualizer:
    """Utility for rendering ASCII memory layout and parameter state transitions."""

    full_char = "█"
    empty_char = "░"

    @classmethod
    def render_shard_bar(
        cls,
        rank: int,
        world_size: int,
        is_gathered: bool,
        bar_width: int = 16,
    ) -> str:
        """Renders a single fixed-width ASCII bar representing memory layout for a rank.

        Examples (world_size=2, bar_width=16):
        - Gathered full tensor: [████████████████]
        - Rank 0 local shard:   [████████░░░░░░░░]
        - Rank 1 local shard:   [░░░░░░░░████████]
        """
        if is_gathered:
            return f"[{cls.full_char * bar_width}]"

        if world_size <= 0 or bar_width <= 0:
            return f"[{cls.empty_char * bar_width}]"

        # Calculate character slots for this rank's shard slice
        slot_size = bar_width / world_size
        start_slot = int(round(rank * slot_size))
        end_slot = int(round((rank + 1) * slot_size))

        # Clamp slots to bounds
        start_slot = max(0, min(bar_width, start_slot))
        end_slot = max(start_slot, min(bar_width, end_slot))

        left_pad = cls.empty_char * start_slot
        shard_segment = cls.full_char * (end_slot - start_slot)
        right_pad = cls.empty_char * (bar_width - end_slot)

        return f"[{left_pad}{shard_segment}{right_pad}]"

    @classmethod
    def render_transition_bars(
        cls,
        size_before: torch.Size,
        size_after: torch.Size,
        rank: int,
        world_size: int,
        bar_width: int = 16,
    ) -> tuple[str, str]:
        """Calculates transition state bars (before -> after) based on shape changes."""
        numel_before = size_before.numel()
        numel_after = size_after.numel()

        # Determine if transition is gathering (sharded -> full) or freeing/scattering (full -> sharded)
        if numel_before < numel_after:
            # Gathering: before is sharded, after is full
            bar_before = cls.render_shard_bar(rank, world_size, is_gathered=False, bar_width=bar_width)
            bar_after = cls.render_shard_bar(rank, world_size, is_gathered=True, bar_width=bar_width)
        elif numel_before > numel_after:
            # Freeing / Reduce-Scatter: before is full, after is sharded
            bar_before = cls.render_shard_bar(rank, world_size, is_gathered=True, bar_width=bar_width)
            bar_after = cls.render_shard_bar(rank, world_size, is_gathered=False, bar_width=bar_width)
        else:
            # Same shape: maintain state based on size comparison relative to world size assumption
            is_gathered = True
            bar_before = cls.render_shard_bar(rank, world_size, is_gathered=is_gathered, bar_width=bar_width)
            bar_after = bar_before

        return bar_before, bar_after


class RankLogger:
    """Distributed rank-aware logger with ASCII lifecycle visualization."""

    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.last_layer_prefix: str | None = None
        self.log_hooks: bool = False

    def log(self, msg: str) -> None:
        print(self.format_log(msg), flush=True)

    def log_rank0(self, msg: str) -> None:
        if self.rank == 0:
            print(msg, flush=True)

    def format_log(self, msg: str) -> str:
        return f"[Rank {self.rank}/{self.world_size}] {msg}"

    def log_ordered(self, msg: str, format_log: bool = True) -> None:
        """Guarantees sequentially ordered stdout across all ranks via process barriers."""
        if not dist.is_initialized():
            if msg.strip():
                self.log(msg)
            return

        for r in range(self.world_size):
            if self.rank == r and msg.strip():
                formatted = "\n".join(self.format_log(line) if format_log else line for line in msg.splitlines())
                print(formatted, flush=True)

            dist.barrier()

    def _extract_layer_prefix(self, param_name: str) -> str:
        parts = param_name.split(".")
        return parts[0] if len(parts) > 1 else param_name

    def reset_step(self) -> None:
        """Resets layer boundary tracking at training step boundaries."""
        self.last_layer_prefix = None

    def log_transition(
        self,
        event_name: str,
        parameter_name: str,
        size_before: torch.Size,
        size_after: torch.Size,
    ) -> None:
        """Logs parameter shape transitions and visual ASCII shard state."""
        if not self.log_hooks:
            return

        # Render memory layout transition
        before_bar, after_bar = ASCIIVisualizer.render_transition_bars(
            size_before=size_before,
            size_after=size_after,
            rank=self.rank,
            world_size=self.world_size,
        )

        current_prefix = self._extract_layer_prefix(parameter_name)

        # Print section dividers when moving to a new module layer on rank 0
        if self.last_layer_prefix is None or current_prefix != self.last_layer_prefix:
            self.log_rank0(f"\n {' ' * 51} {'-' * 5} {current_prefix} {'-' * 5}\n")

        self.last_layer_prefix = current_prefix

        shape_str = f"{tuple(size_before)} -> {tuple(size_after)}"

        msg = (
            f"{'-' * 125}\n"
            f"Rank {self.rank} | {parameter_name:<25} | {event_name:<15} | "
            f"State: {before_bar} -> {after_bar} | Shape: {shape_str}\n"
        )
        self.log_ordered(msg, format_log=False)


logger = RankLogger()

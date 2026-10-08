"""
Rank-aware logger and ASCII visualization utilities for Educational FSDP.

Provides:
1. `ASCIIVisualizer`: Renders memory shard layouts (e.g., `[████████░░░░░░░░]`).
2. `RankLogger`: Handles distributed rank logging, barriers, and state transitions.
"""

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
        shape_before: torch.Size,
        shape_after: torch.Size,
        rank: int,
        world_size: int,
        bar_width: int = 16,
    ) -> tuple[str, str]:
        """Calculates transition state bars (before -> after) based on shape changes."""
        numel_before = shape_before.numel()
        numel_after = shape_after.numel()

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
            # Equal shape: maintain full state
            bar_before = cls.render_shard_bar(rank, world_size, is_gathered=True, bar_width=bar_width)
            bar_after = bar_before

        return bar_before, bar_after


class RankLogger:
    """Distributed rank-aware logger with ASCII lifecycle visualization."""

    # Formatting layout constants
    LINE_WIDTH = 130
    LAYER_NAME_PREFIX_WIDTH = 10
    PARAM_COLUMN_WIDTH = 20
    EVENT_COLUMN_WIDTH = 15

    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.last_layer_prefix: str | None = None
        self.log_hooks: bool = False

    def log(self, message: str) -> None:
        """Logs message with rank prefix."""
        print(self.add_prefix_rank(message), flush=True)

    def log_rank0(self, message: str) -> None:
        """Logs message only on process rank 0."""
        if self.rank == 0:
            print(message, flush=True)
        if dist.is_initialized():
            dist.barrier()

    def log_section_header(self, title: str) -> None:
        """Logs a centered major section header (e.g. 'Forward pass') on Rank 0."""
        banner = f" {title} ".center(self.LINE_WIDTH, "=")
        self.log_rank0(f"\n\n{banner}\n")

    def log_header(self, title: str, symbol: str = "#", width: int = 30) -> None:
        """Logs a prominent block header (e.g. 'LOSS VALUES') on Rank 0."""
        title = f" {title} ".center(width, symbol)
        self.log_rank0(f"\n\n{title}\n")

    def log_divider(self, symbol: str = "#", width: int = 30) -> None:
        """Logs a visual divider line on Rank 0."""
        self.log_rank0(f"\n{symbol * width}")

    def add_prefix_rank(self, message: str) -> str:
        """Formats string with rank prefix."""
        return f"[Rank {self.rank}/{self.world_size}] {message}"

    def log_ordered(self, message: str, add_rank_info: bool = True) -> None:
        """Guarantees sequentially ordered stdout across all ranks via process barriers."""
        if not dist.is_initialized():
            if message.strip():
                self.log(message)
            return

        for rank in range(self.world_size):
            if self.rank == rank and message.strip():
                formatted = "\n".join(
                    self.add_prefix_rank(line) if add_rank_info else line for line in message.splitlines()
                )
                print(formatted, flush=True)

            dist.barrier()

    def _extract_layer_prefix(self, parameter_name: str) -> str:
        """Extracts top-level layer prefix from parameter FQN (e.g., 'layer1' from 'layer1.weight')."""
        prefix, _, _ = parameter_name.partition(".")
        return prefix if prefix else parameter_name

    def reset_step(self) -> None:
        """Resets layer boundary tracking at training step boundaries."""
        self.last_layer_prefix = None

    def log_transition(
        self,
        event_name: str,
        parameter_name: str,
        shape_before: torch.Size,
        shape_after: torch.Size,
    ) -> None:
        """Logs parameter shape transitions and visual ASCII shard state."""
        if not self.log_hooks:
            return

        # Render memory layout transition
        before_bar, after_bar = ASCIIVisualizer.render_transition_bars(
            shape_before=shape_before,
            shape_after=shape_after,
            rank=self.rank,
            world_size=self.world_size,
        )

        current_prefix = self._extract_layer_prefix(parameter_name)

        # Print section dividers when moving to a new module layer on rank 0
        if current_prefix != self.last_layer_prefix:
            layer_name_text = f" {current_prefix} ".center(len(current_prefix) + self.LAYER_NAME_PREFIX_WIDTH * 2, "─")
            layer_name_text = f" {layer_name_text} ".center(self.LINE_WIDTH, " ")
            layer_name_text = f"\n{layer_name_text}\n"
            self.log_rank0(layer_name_text)
        self.last_layer_prefix = current_prefix

        # Print entries to a table
        log_entry_divider = "─" if self.rank == 0 else "⁻"
        shape_str = f"{tuple(shape_before)} -> {tuple(shape_after)}"

        msg = (
            f"{log_entry_divider * self.LINE_WIDTH}\n"
            f"Rank {self.rank} │ {parameter_name:<{self.PARAM_COLUMN_WIDTH}} │ "
            f"{event_name:<{self.EVENT_COLUMN_WIDTH}} │ "
            f"Shard: {before_bar} -> {after_bar} │ "
            f"Shape: {shape_str}\n"
        )

        self.log_ordered(msg, add_rank_info=False)
        self.log_rank0(f"{'─' * self.LINE_WIDTH}\n")


logger = RankLogger()

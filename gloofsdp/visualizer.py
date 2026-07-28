class ASCIIVisualizer:
    @staticmethod
    def render_shard_bar(
        total_elements: int,
        active_range: tuple[int, int] | None = None,
        is_gathered: bool = False,
        bar_width: int = 16,
    ) -> str:
        """
        Renders a fixed-width ASCII bar showing active vs sharded memory layout.

        Examples:
        - Gathered full tensor: [████████████████]
        - Rank 0 local shard:   [████████░░░░░░░░]
        - Rank 1 local shard:   [░░░░░░░░████████]
        """
        if is_gathered or active_range is None:
            return f"[{'█' * bar_width}]"

        start, end = active_range
        if total_elements <= 0:
            return f"[{'░' * bar_width}]"

        # Map character slots to tensor index ranges
        start_slot = int((start / total_elements) * bar_width)
        end_slot = int((end / total_elements) * bar_width)

        # Clamp slots within bounds
        start_slot = max(0, min(bar_width, start_slot))
        end_slot = max(start_slot, min(bar_width, end_slot))

        bar = []
        for i in range(bar_width):
            if start_slot <= i < end_slot:
                bar.append("█")
            else:
                bar.append("░")

        return f"[{''.join(bar)}]"

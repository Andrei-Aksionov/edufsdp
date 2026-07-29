import torch


class CPUMemoryTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sharded_param_elements: int = 0
        self.gathered_param_elements: int = 0
        self.gradient_elements: int = 0
        self.peak_total_elements: int = 0

    def track_sharded_param(self, tensor: torch.Tensor) -> None:
        if tensor is not None:
            self.sharded_param_elements += tensor.numel()
            self._update_peak()

    def track_gathered_param(self, tensor: torch.Tensor) -> None:
        if tensor is not None:
            self.gathered_param_elements += tensor.numel()
            self._update_peak()

    def untrack_gathered_param(self, tensor: torch.Tensor) -> None:
        if tensor is not None:
            self.gathered_param_elements = max(0, self.gathered_param_elements - tensor.numel())

    def track_gradient(self, tensor: torch.Tensor) -> None:
        if tensor is not None:
            self.gradient_elements += tensor.numel()
            self._update_peak()

    def untrack_gradient(self, tensor: torch.Tensor) -> None:
        if tensor is not None:
            self.gradient_elements = max(0, self.gradient_elements - tensor.numel())

    def _update_peak(self) -> None:
        current_total = self.sharded_param_elements + self.gathered_param_elements + self.gradient_elements
        if current_total > self.peak_total_elements:
            self.peak_total_elements = current_total

    def get_summary(self, element_size_bytes: int = 4) -> str:
        """Returns a formatted snapshot of current vs peak memory allocations."""
        total_curr = self.sharded_param_elements + self.gathered_param_elements + self.gradient_elements

        curr_kb = (total_curr * element_size_bytes) / 1024
        peak_kb = (self.peak_total_elements * element_size_bytes) / 1024

        return (
            f"=== CPU Memory Snapshot ===\n"
            f"Parameter shards:    {self.sharded_param_elements:>6} elements\n"
            f"Gathered parameters: {self.gathered_param_elements:>6} elements\n"
            f"Gradients:           {self.gradient_elements:>6} elements\n"
            f"-----------------------------------\n"
            f"Current Total:       {total_curr:>6} elements ({curr_kb:.2f} KB)\n"
            f"Peak Memory:         {self.peak_total_elements:>6} elements ({peak_kb:.2f} KB)"
        )


# Global singleton instance for memory tracking
memory_tracker = CPUMemoryTracker()

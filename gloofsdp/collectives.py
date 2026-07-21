import torch
import torch.distributed as dist


def all_gather_parameter(local_shard: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Gathers local shards from all ranks to reconstruct the full parameter.
    """
    world_size = dist.get_world_size()

    # Create a destination list to receive shards from all ranks
    shard_list = [torch.empty_like(local_shard) for _ in range(world_size)]

    # Synchronous communication
    dist.all_gather(shard_list, local_shard)

    # Concatenate along the flat dimension and reconstruct original shape
    flat_full = torch.cat(shard_list, dim=0)

    # Trim padding if original size wasn't perfectly divisible
    total_elements = original_shape.numel()
    flat_full = flat_full[:total_elements]

    return flat_full.view(original_shape)

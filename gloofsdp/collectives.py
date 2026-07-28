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


def reduce_scatter_tensor(full_tensor_grad: torch.Tensor, shard_size: int) -> torch.Tensor:
    """
    Reduces full gradients across ranks (sum) and scatters the result
    so each rank gets its corresponding local gradient shard.
    """
    world_size: int = dist.get_world_size()

    flat_grad: torch.Tensor = full_tensor_grad.flatten()

    # Pad if the tensor size isn't evenly divisible by world_size
    padding_needed: int = (world_size - (flat_grad.numel() % world_size)) % world_size
    if padding_needed > 0:
        flat_grad = torch.cat([flat_grad, torch.zeros(padding_needed, device=flat_grad.device)])

    # Split the reduced gradient into equal chunks for scattering
    input_list: list[torch.Tensor] = list(torch.chunk(flat_grad, world_size))
    output_shard: torch.Tensor = torch.empty(shard_size, device=full_tensor_grad.device, dtype=full_tensor_grad.dtype)

    # Sum gradients across ranks and scatter local piece
    dist.reduce_scatter(output_shard, input_list, op=dist.ReduceOp.SUM)

    return output_shard

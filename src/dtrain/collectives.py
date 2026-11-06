import torch
import torch.distributed as dist


def _neighbors() -> tuple[int, int, int, int]:
    rank, world = dist.get_rank(), dist.get_world_size()
    return rank, world, (rank + 1) % world, (rank - 1) % world


def _chunks(tensor: torch.Tensor, world: int) -> torch.Tensor:
    if tensor.numel() % world != 0:
        raise ValueError(f"numel {tensor.numel()} is not divisible by world size {world}")
    if not tensor.is_contiguous():
        raise ValueError("ring collectives operate in place and need a contiguous tensor")
    return tensor.view(world, -1)


def _exchange(outgoing: torch.Tensor, incoming: torch.Tensor, right: int, left: int) -> None:
    for request in (dist.isend(outgoing, right), dist.irecv(incoming, left)):
        request.wait()

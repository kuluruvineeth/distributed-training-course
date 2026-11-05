from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=10))
    rank, world = dist.get_rank(), dist.get_world_size()

    x = torch.tensor([float(rank + 1)])
    if rank != 2:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] all_reduce -> {x.item()}", flush=True)

    mine = torch.tensor([rank * 10])
    out = [torch.zeros_like(mine) for _ in range(world)]
    dist.all_gather(out, mine)
    print(f"[rank {rank}] all_gather -> {[t.item() for t in out]}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

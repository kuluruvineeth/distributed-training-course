import time

import torch
import torch.distributed as dist


def fmt(t):
    if t is None:
        return "-"
    if isinstance(t, list):
        return "[" + ", ".join(fmt(x) for x in t) + "]"
    return str([int(v) for v in t.tolist()])


def table(title, before, after):
    rows = [None] * dist.get_world_size()
    dist.gather_object((fmt(before), fmt(after)), rows if dist.get_rank() == 0 else None, dst=0)
    if dist.get_rank() == 0:
        print(f"\n{title}")
        width = max(len(b) for b, _ in rows)
        for r, (b, a) in enumerate(rows):
            print(f"  rank {r}  {b:<{width}}  ->  {a}")


def main():
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    powers = torch.arange(1, world + 1) * 10**rank

    x = powers.clone()
    before = x.clone()
    dist.broadcast(x, src=0)
    table("broadcast(x, src=0)", before, x)

    x = powers.clone()
    before = x.clone()
    dist.reduce(x, dst=0)
    table("reduce(x, dst=0)   only rank 0's result is defined", before, x if rank == 0 else None)

    x = powers.clone()
    before = x.clone()
    dist.all_reduce(x)
    table("all_reduce(x)", before, x)

    x = torch.tensor([rank * 10])
    out = [torch.zeros_like(x) for _ in range(world)] if rank == 0 else None
    dist.gather(x, out, dst=0)
    table("gather(x, dst=0)", x, out)

    out = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(out, x)
    table("all_gather(out, x)", x, out)

    pieces = [torch.tensor([r * 10]) for r in range(world)] if rank == 0 else None
    x = torch.zeros(1, dtype=torch.long)
    dist.scatter(x, pieces, src=0)
    table("scatter(x, pieces, src=0)", pieces, x)

    x = powers.clone()
    out = torch.zeros(1, dtype=x.dtype)
    dist.reduce_scatter_single(out, x)
    table("reduce_scatter_single(out, x)", x, out)

    x = torch.arange(world) + 10 * rank
    out = torch.zeros_like(x)
    dist.all_to_all_single(out, x)
    table("all_to_all_single(out, x)", x, out)

    send = torch.tensor([rank])
    recv = torch.zeros(1, dtype=torch.long)
    reqs = [dist.isend(send, (rank + 1) % world), dist.irecv(recv, (rank - 1) % world)]
    for r in reqs:
        r.wait()
    table("isend right, irecv left", send, recv)

    start = time.perf_counter()
    time.sleep(0.2 * rank)
    dist.barrier()
    waited = time.perf_counter() - start
    rows = [None] * world
    dist.gather_object(waited, rows if rank == 0 else None, dst=0)
    if rank == 0:
        print("\nbarrier()   rank r sleeps 0.2·r s first")
        for r, w in enumerate(rows):
            print(f"  rank {r}  left the barrier after {w:.2f} s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

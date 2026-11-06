import torch
import torch.distributed as dist


def ring_all_reduce(tensor):
    rank, world = dist.get_rank(), dist.get_world_size()
    assert tensor.numel() % world == 0, "size must divide evenly by world"

    chunks = tensor.view(world, -1)
    right, left = (rank + 1) % world, (rank - 1) % world
    inbox = torch.empty_like(chunks[0])

    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        reqs = [dist.isend(chunks[send_idx], right), dist.irecv(inbox, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx] += inbox

    for step in range(world - 1):
        send_idx = (rank + 1 - step) % world
        recv_idx = (rank - step) % world
        reqs = [dist.isend(chunks[send_idx], right), dist.irecv(inbox, left)]
        for r in reqs:
            r.wait()
        chunks[recv_idx].copy_(inbox)


def main():
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    x = torch.arange(1, world + 1, dtype=torch.float64) * 10**rank
    expected = x.clone()
    dist.all_reduce(expected)

    ring_all_reduce(x)
    print(
        f"[rank {rank}] mine={x.tolist()} pytorch={expected.tolist()} "
        f"match={torch.equal(x, expected)}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

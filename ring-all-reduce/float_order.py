import torch
import torch.distributed as dist

from dtrain import ring_all_reduce


def main():
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    torch.manual_seed(rank)
    x = torch.randn(world * 1000)
    expected = x.clone()
    dist.all_reduce(expected)
    ring_all_reduce(x)

    diff = (x - expected).abs()
    failing = ~torch.isclose(x, expected)
    if rank == 0:
        print(f"world={world}")
        print(f"  torch.equal                       {torch.equal(x, expected)}")
        print(f"  largest difference                {diff.max().item():.2e}")
        print(f"  torch.allclose (default atol)     {torch.allclose(x, expected)}")
        print(f"  elements failing                  {failing.sum().item()} of {x.numel()}")
        print(f"  largest |sum| among the failures  {expected[failing].abs().max().item():.3f}")
        print(f"  torch.allclose (atol=1e-5)        {torch.allclose(x, expected, atol=1e-5)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

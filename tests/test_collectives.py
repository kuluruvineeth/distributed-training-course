import pytest
import torch
import torch.distributed as dist

from dtrain import ring_all_gather, ring_all_reduce, ring_reduce_scatter
from tests.harness import run_distributed

WORLD_SIZES = [2, 3, 4, 8]


def _powers_of_ten_match(rank, world):
    x = torch.arange(1, world + 1, dtype=torch.float64) * 10**rank
    expected = x.clone()
    dist.all_reduce(expected)
    ring_all_reduce(x)
    assert torch.equal(x, expected)


def _random_floats_match(rank, world):
    torch.manual_seed(rank)
    x = torch.randn(world * 1000)
    expected = x.clone()
    dist.all_reduce(expected)
    ring_all_reduce(x)
    torch.testing.assert_close(x, expected, rtol=1e-5, atol=1e-5)
    replicas = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(replicas, x)
    for replica in replicas:
        assert torch.equal(replica, x)


def _reduce_scatter_owns_one_full_chunk(rank, world):
    x = torch.full((world * 3,), float(rank + 1))
    owned = ring_reduce_scatter(x)
    assert owned == (rank + 1) % world
    total = world * (world + 1) / 2
    assert torch.equal(x.view(world, -1)[owned], torch.full((3,), total))


def _all_gather_copies_owned_chunks(rank, world):
    x = torch.zeros(world * 2)
    owned = (rank + 1) % world
    x.view(world, -1)[owned] = owned
    ring_all_gather(x, owned)
    assert torch.equal(x, torch.arange(world).repeat_interleave(2).float())


@pytest.mark.parametrize("world", WORLD_SIZES, ids=lambda w: f"N{w}")
@pytest.mark.parametrize(
    "check",
    [
        _powers_of_ten_match,
        _random_floats_match,
        _reduce_scatter_owns_one_full_chunk,
        _all_gather_copies_owned_chunks,
    ],
    ids=["pow10", "randn", "reduce_scatter", "all_gather"],
)
def test_ring(world, check):
    run_distributed(world, check)


def _rejects_uneven_split(rank, world):
    with pytest.raises(ValueError, match="not divisible"):
        ring_all_reduce(torch.ones(world + 1))


def test_rejects_uneven_split():
    run_distributed(2, _rejects_uneven_split)

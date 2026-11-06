# Ring all-reduce

Every rank has a different gradient and needs the sum. If every rank sends to rank 0, rank 0 moves (N − 1) × D in each direction while the other links sit idle. Can every rank share that work?

`ring_all_reduce.py` builds all-reduce from `isend` and `irecv`, then compares the result with `dist.all_reduce` on four CPU processes. No GPU is needed.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run torchrun --nproc-per-node=4 --master-port=29500 ring-all-reduce/ring_all_reduce.py
```

On macOS, put `GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1` in front of `uv run`. Every rank should print `match=True`:

```
[rank 0] mine=[1111.0, 2222.0, 3333.0, 4444.0] pytorch=[1111.0, 2222.0, 3333.0, 4444.0] match=True
```

## The ring

Split the tensor into N chunks. On every step, each rank sends one chunk to the right and receives one from the left.

| Phase | Steps | Received chunk | Result |
|---|---|---|---|
| Reduce-scatter | N − 1 | added (`+=`) | rank r holds the full sum of chunk (r + 1) % N |
| All-gather | N − 1 | copied (`.copy_`) | every rank holds every summed chunk |

Rank r starts with `[1, 2, 3, 4] × 10^r`, so each digit of a partial sum shows which ranks have added to it. Chunk 2 grows `300 → 3300 → 3303 → 3333`. The demo uses float64: with eight ranks these sums pass 2^24, where float32 can no longer store every integer.

The ring computes a sum. Averaging gradients means dividing by N afterwards.

`src/dtrain/collectives.py` keeps the two phases as separate functions, `ring_reduce_scatter` and `ring_all_gather`. The tests compare them with `torch.distributed` on 2, 3, 4 and 8 ranks:

```bash
uv run pytest -q tests/test_collectives.py
```

## Cost

Each rank sends 2(N − 1) chunks of size D/N: always less than 2D.

| N | rank 0, everyone sends to it | ring, every rank |
|---|---|---|
| 4 | 3D | 1.5D |
| 8 | 7D | 1.75D |
| 1024 | 1023D | ≈ 2D |

Bandwidth per rank stays flat as N grows. The number of steps, 2(N − 1), does not, so latency still matters for small tensors. That is one reason NCCL also has tree algorithms.

## Floating point

```bash
uv run torchrun --nproc-per-node=8 --master-port=29500 ring-all-reduce/float_order.py
```

```
world=8
  torch.equal                       False
  largest difference                1.43e-06
  torch.allclose (default atol)     False
  elements failing                  15 of 8000
  largest |sum| among the failures  0.022
  torch.allclose (atol=1e-5)        True
```

Both results are correct sums of the same random floats, added in a different order. In float32, `(1e8 + 1) - 1e8` is `0` but `(1e8 - 1e8) + 1` is `1`.

Default `allclose` allows `1e-8 + 1e-5 × |expected|`. Every failing element is a sum that cancelled to nearly zero, so the relative part gives it almost no room. Pick `atol` from the size of the inputs, not the output. The ranks still agree with each other exactly, because all-gather only copies.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/ring-all-reduce/ring-all-reduce.excalidraw) to follow the chunks step by step.

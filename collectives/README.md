# How GPUs communicate

Each process has its own tensors. How do we copy one tensor to everyone, add values across processes, or exchange different pieces?

`collectives.py` runs four processes on CPU with Gloo and prints each rank's tensors before and after the calls. No GPU is needed.

## Run it

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run torchrun --nproc-per-node=4 --master-port=29500 collectives/collectives.py
```

On macOS, select the loopback interface for this local demo:

```bash
GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1 uv run torchrun --nproc-per-node=4 --master-port=29500 collectives/collectives.py
```

| Operation | Result |
|---|---|
| Broadcast | Every rank gets the root's tensor. |
| Scatter | Each rank gets one piece from the root. |
| Reduce | The root gets the sum. Other ranks' results are undefined. |
| Gather | The root gets every rank's piece, without adding. |
| All-reduce | Every rank gets the sum. |
| All-gather | Every rank gets every piece. |
| Reduce-scatter | Each rank keeps one slice of the sum. |
| All-to-all | Each rank gets a different piece from every rank. |
| Send / receive | A tensor moves between two ranks. |
| Barrier | All ranks wait until everyone arrives. |

For the sum examples, rank `r` starts with `[1, 2, 3, 4] × 10**r`. All-reduce gives every rank `[1111, 2222, 3333, 4444]`; reduce-scatter gives them `[1111]`, `[2222]`, `[3333]`, and `[4444]` respectively. All-gather can assemble those slices into the full sum again.

## Make it fail

```bash
uv run torchrun --nproc-per-node=4 --master-port=29500 collectives/deadlock.py
```

Use the same environment variables on macOS. This script is supposed to fail: rank 2 skips `all_reduce` and calls `all_gather` while the other ranks are still trying to reduce. The process group has a ten-second timeout.

Ranks in a process group must call matching collectives in the same order. Receive buffers also need the appropriate shape and dtype. For point-to-point communication, the demo posts both send and receive before waiting.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/collectives/collectives.excalidraw) to follow the tensors visually.

# Single-device baseline

A parallel training run can be fast and still wrong, and nothing crashes when it is. How would we know? Train the same model on one device first and keep its loss and gradients as the reference.

These scripts train a toy Llama on CPU, then check that gradient accumulation and activation recomputation change memory and speed without changing the result. No GPU is needed.

## Train

From the repository root, after `uv sync --locked --python 3.12`:

```bash
uv run python single-device/train.py
uv run python single-device/train.py --mbs 2 --grad-acc 4
uv run python single-device/train.py --recompute
```

The model has 123,712 parameters and learns the bytes of this repo's `LICENSE` (vocabulary 256). Each window is `seq + 1` bytes: inputs are `batch[:, :-1]`, targets are `batch[:, 1:]`.

All three runs use the same 8 × 64 tokens per step:

| step | mbs 8 | mbs 2 × grad_acc 4 | mbs 8, `--recompute` |
|---|---|---|---|
| 1 | 5.886 | 5.886 | 5.886 |
| 25 | 3.538 | 3.538 | 3.538 |
| 50 | 2.757 | 2.757 | 2.757 |
| 100 | 2.290 | 2.290 | 2.290 |

Tokens per second varies from run to run on a laptop. The loss should not.

Gradient accumulation divides each micro-batch loss by `grad_acc`, and `backward()` adds into `.grad`, so four small steps give the average over eight sequences. That matches the big batch only when every micro-batch has the same number of tokens, which fixed-length windows guarantee.

The tests check the same thing directly: accumulated gradients match one big batch (within 1e-6 in fp32, 1e-12 in fp64), and recomputation gives identical gradients.

```bash
uv run pytest -q tests/test_train.py tests/test_model.py
```

## Memory

```bash
uv run python single-device/memory.py
```

```
Llama 3 8B: 8,030,261,248 parameters
  fp32                              128.5 GB
  bf16 mixed · weights                16.1 GB
  bf16 mixed · gradients              16.1 GB
  bf16 mixed · master weights         32.1 GB
  bf16 mixed · optimizer              64.2 GB
  bf16 mixed · fp32 gradients         32.1 GB
  bf16 mixed · total                 160.6 GB
  activations, mbs=1, seq=4096        104.2 GB
```

With bf16 weights and gradients, fp32 master weights, Adam's two fp32 moments and an fp32 gradient buffer for accumulation, that's 2 + 2 + 4 + 8 + 4 = 20 bytes per parameter. Plain fp32 is 16. Mixed precision speeds up the matmuls and shrinks activations; it doesn't shrink the training state. Llama 3 70B needs 1.4 TB of it before a single activation.

The activation line uses the estimate from [Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198), `L · seq · mbs · h · (34 + 5 · a · seq / h)` bytes, which assumes the attention scores are stored.

## Activations, measured

```bash
uv run python single-device/activations.py
```

Bytes autograd keeps for backward on a 4-layer model, weights excluded:

```
   seq        stored     recompute
   128       2.98 MB       0.38 MB
   256       5.96 MB       0.76 MB
   512      11.92 MB       1.51 MB
  1024      23.83 MB       3.03 MB
```

This grows linearly with sequence length. PyTorch's `scaled_dot_product_attention` doesn't keep the `seq × seq` scores, so the quadratic term in the estimate is already gone. Recomputation keeps only each layer's input and runs the layer again during backward: about 8× less memory here, for one extra forward pass. This counts saved tensors only, not total peak memory.

[Download the board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/single-device-baseline/single-device-baseline.excalidraw) to follow the diagrams.

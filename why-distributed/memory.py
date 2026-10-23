import math

import torch
from models import GPU_MEMORY, MODELS

from dtrain.memory import num_params

GB = 1e9
BYTES = [
    ("bf16 weights", 2, "the copy the forward and backward matmuls read"),
    ("bf16 gradients", 2, "what backward writes, one per weight"),
    ("fp32 master weights", 4, "bf16 cannot hold small updates, shown below"),
    ("Adam first moment", 4, "running average of the gradient, fp32"),
    ("Adam second moment", 4, "running average of the squared gradient, fp32"),
    ("fp32 gradient buffer", 4, "summing micro-batches in bf16 drops small values, shown below"),
]


def lost_update():
    w = torch.tensor(1.0, dtype=torch.bfloat16)
    step = 1e-3
    print("why fp32 master weights")
    print(f"  bf16  1.0 - {step} = {(w - step).item()}   the update disappears")
    print(f"  fp32  1.0 - {step} = {(w.float() - step).item():.4f}")
    eps = torch.finfo(torch.bfloat16).eps
    print(f"  bf16 keeps 8 bits of precision: the next value above 1.0 is {1 + eps}")


def stalled_sum():
    bf16 = torch.tensor(0.0, dtype=torch.bfloat16)
    fp32 = torch.tensor(0.0)
    for _ in range(1000):
        bf16 += 1e-3
        fp32 += 1e-3
    print("\nwhy an fp32 gradient buffer")
    print("  add 0.001 a thousand times")
    print(f"  bf16  {bf16.item()}   near 0.5, bf16 values are 0.0039 apart: + 0.001 rounds back")
    print(f"  fp32  {fp32.item():.4f}")


def state():
    total = sum(b for _, b, _ in BYTES)
    print("\nbytes per parameter, bf16 mixed precision with Adam")
    for name, size, why in BYTES:
        print(f"  {name:<22} {size:>2}   {why}")
    print(f"  {'total':<22} {total:>2}")
    print("  The ZeRO paper counts 16: same list without the fp32 gradient buffer.")
    for name, config in MODELS.items():
        params = num_params(config)
        print(f"\n{name}: {params:,} parameters × {total} bytes")
        for part, size, _ in BYTES:
            print(f"  {part:<22} {params * size / GB:>9,.1f} GB")
        print(f"  {'training state':<22} {params * total / GB:>9,.1f} GB")
        gpus = params * total / GPU_MEMORY
        print(f"  80 GB GPUs to hold it  {gpus:>9.2f}  → at least {math.ceil(gpus)}")


def activations():
    config = MODELS["Llama 3 8B"]
    layers, h, heads = config.num_hidden_layers, config.hidden_size, config.num_attention_heads
    print("\nactivations: tensors forward keeps for backward")
    print("  per layer, in bf16, for one sequence: seq · h · (34 + 5 · heads · seq / h) bytes")
    print("  34 · seq · h: inputs of the matmuls, norms, activation functions and dropout masks")
    print(
        "  5 · heads · seq²: attention scores, softmax output and dropout mask, seq × seq per head"
    )
    print("  (Korthikanti et al. 2022, for a GPT-style layer; Llama's SwiGLU changes the 34)")
    print(f"\n  Llama 3 8B, {layers} layers, micro-batch of 1 sequence")
    print(f"  {'seq':>6}  {'scores kept':>13}  {'FlashAttention':>15}")
    for seq in (1024, 4096, 8192):
        base = layers * seq * h * 34
        scores = layers * 5 * heads * seq * seq
        print(f"  {seq:>6}  {(base + scores) / GB:>10.1f} GB  {base / GB:>12.1f} GB")
    print("  FlashAttention never stores the seq × seq matrix: backward recomputes it in tiles")


def main():
    lost_update()
    stalled_sum()
    state()
    activations()


if __name__ == "__main__":
    main()

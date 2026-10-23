import torch
from models import GPU_PEAK_FLOPS, MFU, MODELS, REPORTED_GPU_HOURS, SEQ_LENGTH, TOKENS
from torch.utils.flop_counter import FlopCounterMode

from dtrain.flops import (
    attention_flops_per_token,
    embedding_params,
    matmul_params,
    norm_params,
)
from dtrain.memory import num_params
from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn


def measured(fn) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def check_on_small_model():
    config = LlamaConfig(max_position_embeddings=128)
    seq, batch = 128, 2
    tokens = seq * batch
    model = Llama(config)
    ids = torch.randint(0, config.vocab_size, (batch, seq + 1))
    total = measured(lambda: loss_fn(model(ids[:, :-1]), ids[:, 1:]).backward())
    print(f"\n5. Checked with torch.utils.flop_counter on a {num_params(config):,}-parameter Llama")
    print(f"   {'matmul FLOPs per token, measured':<42} {total // tokens:>10,}")
    print(
        f"   {f'6 × {matmul_params(config):,} matmul weights':<42} {6 * matmul_params(config):>10,}"
    )

    h, heads = config.hidden_size, config.num_attention_heads
    q, k, v = (torch.randn(batch, heads, seq, h // heads, requires_grad=True) for _ in range(3))

    def attention():
        scores = torch.softmax(q @ k.transpose(-1, -2) / (h // heads) ** 0.5, dim=-1)
        (scores @ v).sum().backward()

    print(
        f"   {'attention per layer per token, measured':<42} {measured(attention) // tokens:>10,}"
    )
    print(f"   {f'12 × h {h} × seq {seq}':<42} {12 * h * seq:>10,}")


def main():
    config = MODELS["Llama 3 8B"]
    params = num_params(config)
    matmul = matmul_params(config)
    attention = attention_flops_per_token(config, SEQ_LENGTH)
    layers, h = config.num_hidden_layers, config.hidden_size

    print("1. One matrix multiply")
    print("   (m × k) @ (k × n): each of the m·n outputs takes k multiplies and k adds")
    print("   → 2·m·k·n FLOPs. Per token, each matmul weight costs 2: one multiply, one add")

    print("\n2. Forward pass, per token, Llama 3 8B")
    print(f"   all parameters                     {params:>16,}")
    print(f"   − embedding (a lookup, no math)    {embedding_params(config):>16,}")
    print(f"   − RMSNorm weights (elementwise)    {norm_params(config):>16,}")
    print(f"   = weights used in matmuls          {matmul:>16,}")
    print(f"   forward: 2 × {matmul:,} = {2 * matmul / 1e9:.1f} GFLOPs per token")

    print("\n3. Backward pass = 2 × forward")
    print("   y = x·W goes back through two matmuls of the same size:")
    print("   dx = dy·Wᵀ, to keep backpropagating, and dW = xᵀ·dy, to update W")
    print(f"   forward + backward: 6 × {matmul:,} = {6 * matmul / 1e9:.1f} GFLOPs per token")

    print("\n4. Attention adds work that has no weights")
    print("   each token's query is compared with every key in the sequence: 2·h·seq per layer")
    print("   and mixes the values with those scores: 2·h·seq per layer")
    print("   4·h·seq forward, × 3 with backward = 12·h·seq per layer per token")
    print(f"   seq {SEQ_LENGTH}: 12 × {layers} × {h} × {SEQ_LENGTH} = {attention / 1e9:.1f} GFLOPs")
    print(f"   that is +{attention / (6 * matmul):.0%} on top of the matmuls")
    print("   (counted without causal masking, the usual convention; masking can skip about half)")

    check_on_small_model()

    per_token = 6 * matmul + attention
    rule = 6 * params * TOKENS
    print(f"\n6. The whole run, {TOKENS / 1e12:.0f}T tokens")
    print(f"   rule of thumb 6·N·D:  6 × {params / 1e9:.2f}e9 × {TOKENS:.1e} = {rule:.2e} FLOPs")
    step = per_token * TOKENS
    print(f"   step by step:         {per_token / 1e9:.1f}e9 × {TOKENS:.1e} = {step:.2e}")
    print("   The rule counts the embedding, which does no math, and leaves out attention.")
    print(f"   At seq {SEQ_LENGTH} the step-by-step count is {step / rule - 1:.0%} higher.")
    print("   The time below uses the rule, the standard convention, so it stays a floor.")

    effective = GPU_PEAK_FLOPS * MFU
    seconds = rule / effective
    hours = seconds / 3600
    print("\n7. FLOPs to time")
    print(f"   H100 bf16 dense peak             {GPU_PEAK_FLOPS / 1e12:.0f} TFLOP/s")
    print(f"   MFU {MFU:.0%}: the model's FLOPs run at {effective / 1e12:.0f} TFLOP/s")
    print(f"   {rule:.2e} / {effective:.2e} = {seconds:.2e} s = {hours / 1e6:.2f}M GPU-hours")
    print(f"   one GPU: {hours / 24 / 365:.0f} years")
    for gpus in (1024, 16384):
        print(f"   {gpus:,} GPUs, perfect scaling: {hours / gpus / 24:.1f} days")
    reported = REPORTED_GPU_HOURS["Llama 3 8B"]
    print(f"   Meta reports {reported / 1e6:.2f}M GPU-hours (Llama 3.1 model card)")
    print(f"   {reported / hours:.1f}× this estimate, which leaves out attention and every stall")


if __name__ == "__main__":
    main()

import argparse

import torch
from models import MODELS
from tokenizers import Tokenizer
from torch.utils.flop_counter import FlopCounterMode

from dtrain.flops import matmul_params
from dtrain.model import Llama

TOKENIZER = "NousResearch/Meta-Llama-3-8B"


def dims(shape) -> str:
    return " × ".join(f"{d:,}" for d in shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="What I cannot create, I do not understand.")
    text = parser.parse_args().text
    config = MODELS["Llama 3 8B"]
    h, heads, layers = config.hidden_size, config.num_attention_heads, config.num_hidden_layers

    tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    pieces = [tokenizer.decode([i]) for i in ids]
    t = len(ids)

    print("1. Text to tokens, with the Llama 3 tokenizer")
    print(f'   "{text}"')
    width = max(len(repr(p)) for p in pieces) + 2
    print("   token " + "".join(f"{repr(p):>{width}}" for p in pieces))
    print("   id    " + "".join(f"{i:>{width}}" for i in ids))
    print(f"   {t} tokens. The vocabulary has {tokenizer.get_vocab_size():,} entries.")

    with torch.device("meta"):
        model = Llama(config)
    seen = {}

    def record(name):
        def hook(module, inputs, output):
            seen[name] = (tuple(inputs[0].shape[1:]), tuple(output.shape[1:]))

        return hook

    for name, module in model.named_modules():
        if name.startswith("decoder_layers.0.") or name in ("embedding", "final_proj"):
            module.register_forward_hook(record(name))
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.tensor([ids], device="meta"))

    vocab = config.vocab_size
    print(f"\n2. Tokens to vectors: the embedding table, {vocab:,} × {h:,}")
    print(f"   id {ids[0]} picks row {ids[0]}: a vector of {h:,} numbers. A lookup, no arithmetic.")
    print(f"   The sentence becomes a {dims(seen['embedding'][1])} grid: one row per token.")

    print(f"\n3. Layer 0 of {layers}. Shapes from the 8B model run on PyTorch's meta device")
    print(f"   {'':<17} {'input':<13} {'weight (out × in)':<20} {'output':<14} {'FLOPs':>15}")
    layer_flops = 0

    def linear(label, name):
        nonlocal layer_flops
        module = model.get_submodule(f"decoder_layers.0.{name}")
        flops = 2 * t * module.in_features * module.out_features
        layer_flops += flops
        inp, out = seen[f"decoder_layers.0.{name}"]
        weight = f"{module.out_features:,} × {module.in_features:,}"
        print(f"   {label:<17} {dims(inp):<13} {weight:<20} {dims(out):<14} {flops:>15,}")

    def note(label, text, flops=0, kind="elementwise"):
        nonlocal layer_flops
        layer_flops += flops
        shown = f"{flops:,}" if flops else kind
        print(f"   {label:<17} {text:<49} {shown:>15}")

    head_dim = h // heads
    note("input_layernorm", f"rescale each row, {h:,} learned scales")
    linear("q_proj", "attention.q_proj")
    linear("k_proj", "attention.k_proj")
    linear("v_proj", "attention.v_proj")
    kv = config.num_key_value_heads
    note(
        "share k, v",
        f"{kv} key/value heads, each used by {heads // kv} of {heads} query heads",
        kind="copy",
    )
    pair = f"{t} × {head_dim} · ({t} × {head_dim})ᵀ"
    note("scores", f"{heads} heads, {pair} → {heads} × {t} × {t}", 2 * t * t * h)
    note("mix values", f"{heads} × {t} × {t} · {t} × {head_dim} → {t} × {h:,}", 2 * t * t * h)
    linear("out_proj", "attention.out_proj")
    note("+ residual", "add the layer's input back")
    note("post_attn_norm", f"rescale each row, {h:,} learned scales")
    linear("gate_proj", "mlp.gate_proj")
    linear("up_proj", "mlp.up_proj")
    note("silu(gate) × up", "elementwise, on the wide rows")
    linear("down_proj", "mlp.down_proj")
    note("+ residual", "add the layer's input back")
    print(f"   {'layer total':<67} {layer_flops:>15,}")

    final = 2 * t * h * vocab
    print(f"\n4. {layers} layers, then the output")
    inp, out = seen["final_proj"]
    print(f"   final_norm, then final_proj: {dims(inp)} → {dims(out)}")
    print(f"   one score per vocabulary entry for each position: {final:,} FLOPs")
    print(f"   the last row scores every possible token after {pieces[-1]!r}")

    weights = matmul_params(config)
    attention = layers * 4 * t * t * h
    print(f"\n5. The whole forward pass for {t} tokens")
    rows = [
        (f"matmuls    2 × {weights:,} weights × {t} tokens", 2 * weights * t),
        (f"attention  {layers} layers × 4 × {t} × {t} × {h:,}", attention),
        ("total", 2 * weights * t + attention),
        ("PyTorch's FLOP counter, same forward pass", counter.get_total_flops()),
    ]
    for label, value in rows:
        print(f"   {label:<48} {value:>17,}")
    print(
        f"   per token: {(2 * weights * t + attention) / t / 1e9:.1f} GFLOPs, 2 per matmul weight"
    )
    print("   Training adds the backward pass, twice the forward: 6 per weight per token.")


if __name__ == "__main__":
    main()

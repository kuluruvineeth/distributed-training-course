import argparse

import torch
from models import HUGGING_FACE_PARAMS, MODELS

from dtrain.model import Llama


def shape(t: torch.Tensor) -> str:
    return " × ".join(f"{d:,}" for d in t.shape)


def row(name: str, dims: str, count: int, indent: int = 0) -> None:
    print(f"  {' ' * indent}{name:<{30 - indent}} {dims:<22} {count:>16,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="Llama 3 8B")
    name = parser.parse_args().model
    config = MODELS[name]
    with torch.device("meta"):
        model = Llama(config)
    params = dict(model.named_parameters())

    print(f"{name}")
    c = config
    heads = f"{c.num_attention_heads} query heads and {c.num_key_value_heads} key/value heads"
    print(f"hidden {c.hidden_size:,} · {c.num_hidden_layers} layers · {heads} of {c.head_dim}")
    print(f"MLP {c.intermediate_size:,} · vocabulary {c.vocab_size:,}\n")
    print(f"  {'':<30} {'shape (out × in)':<22} {'parameters':>16}")
    row("embedding", shape(params["embedding.weight"]), params["embedding.weight"].numel())
    print("  one decoder layer")
    layer = {n.split(".", 2)[2]: p for n, p in params.items() if n.startswith("decoder_layers.0.")}
    for n, p in layer.items():
        row(n.removesuffix(".weight"), shape(p), p.numel(), indent=2)
    per_layer = sum(p.numel() for p in layer.values())
    row("layer total", "", per_layer, indent=2)
    layers = config.num_hidden_layers
    row(f"{layers} decoder layers", f"{layers} × {per_layer:,}", layers * per_layer)
    row("final_norm", shape(params["final_norm.weight"]), params["final_norm.weight"].numel())
    row("final_proj", shape(params["final_proj.weight"]), params["final_proj.weight"].numel())
    total = sum(p.numel() for p in params.values())
    row("total", "", total)
    print(
        f"\n  Hugging Face safetensors metadata                    {HUGGING_FACE_PARAMS[name]:>16,}"
    )

    groups = {
        "MLP (gate, up, down)": lambda n: ".mlp." in n,
        "attention (q, k, v, out)": lambda n: ".attention." in n,
        "embedding + final_proj": lambda n: n.startswith(("embedding", "final_proj")),
        "RMSNorm weights": lambda n: "norm" in n,
    }
    print("\n  where the parameters are")
    for label, match in groups.items():
        count = sum(p.numel() for n, p in params.items() if match(n))
        share = f"{count / total:.1%}" if count / total >= 0.001 else f"{count / total:.3%}"
        print(f"    {label:<28} {count:>16,}  {share:>7}")

    q = layer["attention.q_proj.weight"].numel()
    k = layer["attention.k_proj.weight"].numel()
    shared = f"{c.num_attention_heads} query heads share {c.num_key_value_heads} key/value heads"
    print(f"\n  k_proj and v_proj are {q // k}× smaller than q_proj: {shared}")


if __name__ == "__main__":
    main()

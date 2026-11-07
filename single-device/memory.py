from dtrain.memory import activation_bytes, num_params, training_bytes
from dtrain.model import LlamaConfig

MODELS = {
    "Llama 3 8B": LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "Llama 3 70B": LlamaConfig(
        vocab_size=128256,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
    ),
}

GB = 1e9


def main():
    for name, config in MODELS.items():
        params = num_params(config)
        print(f"\n{name}: {params:,} parameters")
        fp32 = training_bytes(params, mixed_precision=False)
        mixed = training_bytes(params, mixed_precision=True, fp32_grad_accumulation=True)
        print(f"  fp32                           {sum(fp32.values()) / GB:8.1f} GB")
        for part, size in mixed.items():
            print(f"  bf16 mixed · {part:<18} {size / GB:8.1f} GB")
        print(f"  bf16 mixed · total              {sum(mixed.values()) / GB:8.1f} GB")
        for seq in (1024, 4096, 16384):
            act = activation_bytes(config, seq_length=seq, micro_batch_size=1)
            print(f"  activations, mbs=1, seq={seq:<6}   {act / GB:8.1f} GB")


if __name__ == "__main__":
    main()

import torch
from torch.utils.flop_counter import FlopCounterMode

from dtrain.flops import attention_flops_per_token, matmul_params
from dtrain.memory import num_params
from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn

CONFIG = LlamaConfig(max_position_embeddings=128)
SEQ, BATCH = 128, 2


def count(fn) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def test_matmul_flops_are_six_per_weight_per_token():
    model = Llama(CONFIG)
    ids = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ + 1))
    measured = count(lambda: loss_fn(model(ids[:, :-1]), ids[:, 1:]).backward())
    assert measured == 6 * matmul_params(CONFIG) * BATCH * SEQ


def test_attention_flops_are_twelve_h_seq_per_layer_per_token():
    h, heads = CONFIG.hidden_size, CONFIG.num_attention_heads
    q, k, v = (torch.randn(BATCH, heads, SEQ, h // heads, requires_grad=True) for _ in range(3))

    def attention():
        scores = torch.softmax(q @ k.transpose(-1, -2) / (h // heads) ** 0.5, dim=-1)
        (scores @ v).sum().backward()

    per_layer = attention_flops_per_token(CONFIG, SEQ) // CONFIG.num_hidden_layers
    assert count(attention) == per_layer * BATCH * SEQ


def test_meta_device_count_matches_formula():
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    with torch.device("meta"):
        model = Llama(config)
    assert sum(p.numel() for p in model.parameters()) == num_params(config)

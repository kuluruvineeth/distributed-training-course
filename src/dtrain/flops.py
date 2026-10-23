from dtrain.memory import num_params
from dtrain.model import LlamaConfig


def embedding_params(config: LlamaConfig) -> int:
    return config.vocab_size * config.hidden_size


def norm_params(config: LlamaConfig) -> int:
    return (2 * config.num_hidden_layers + 1) * config.hidden_size


def matmul_params(config: LlamaConfig) -> int:
    return num_params(config) - embedding_params(config) - norm_params(config)


def attention_flops_per_token(config: LlamaConfig, seq_length: int) -> int:
    return 12 * config.num_hidden_layers * config.hidden_size * seq_length


def train_flops_per_token(config: LlamaConfig, seq_length: int) -> int:
    return 6 * matmul_params(config) + attention_flops_per_token(config, seq_length)

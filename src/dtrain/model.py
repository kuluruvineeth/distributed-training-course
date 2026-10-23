from dataclasses import dataclass


@dataclass
class LlamaConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    intermediate_size: int = 172
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

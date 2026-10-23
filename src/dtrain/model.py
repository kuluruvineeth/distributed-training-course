from dataclasses import dataclass

import torch
import torch.nn as nn


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


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)

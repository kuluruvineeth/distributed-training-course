import re

import torch

RENAMES = [
    (r"^model\.", ""),
    (r"^layers\.", "decoder_layers."),
    (r"^embed_tokens", "embedding"),
    (r"\.self_attn\.", ".attention."),
    (r"\.o_proj\.", ".out_proj."),
    (r"^lm_head", "final_proj"),
    (r"^norm\.", "final_norm."),
]


def hf_to_dtrain(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for name, tensor in state_dict.items():
        for pattern, replacement in RENAMES:
            name = re.sub(pattern, replacement, name)
        converted[name] = tensor
    return converted

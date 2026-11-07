from collections.abc import Iterator

import torch


def windows(tokens: torch.Tensor, seq_length: int) -> torch.Tensor:
    count = (tokens.numel() - 1) // seq_length
    return tokens[: count * seq_length + 1].unfold(0, seq_length + 1, seq_length)


def micro_batches(
    tokens: torch.Tensor, seq_length: int, micro_batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    rows = windows(tokens, seq_length)
    usable = rows.size(0) - rows.size(0) % micro_batch_size
    while True:
        for start in range(0, usable, micro_batch_size):
            batch = rows[start : start + micro_batch_size]
            yield batch[:, :-1], batch[:, 1:]

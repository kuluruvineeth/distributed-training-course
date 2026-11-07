from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def train_step(
    model: nn.Module, batches: Iterator[tuple[torch.Tensor, torch.Tensor]], grad_acc: int
) -> float:
    total = 0.0
    for _ in range(grad_acc):
        inputs, targets = next(batches)
        loss = loss_fn(model(inputs), targets) / grad_acc
        loss.backward()
        total += loss.item()
    return total

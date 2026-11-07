import pytest
import torch

from dtrain.data import micro_batches, windows
from dtrain.model import Llama, LlamaConfig
from dtrain.train import train_step

CONFIG = LlamaConfig(max_position_embeddings=32)


def test_windows_shift_by_one():
    tokens = torch.arange(100)
    rows = windows(tokens, 32)
    assert rows.shape == (3, 33)
    inputs, targets = next(micro_batches(tokens, 32, 3))
    assert torch.equal(targets, inputs + 1)


@pytest.mark.parametrize("dtype, atol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
def test_grad_accumulation_matches_one_big_batch(dtype, atol):
    torch.manual_seed(0)
    tokens = torch.randint(
        0, CONFIG.vocab_size, (8 * 32 + 1,), generator=torch.Generator().manual_seed(4)
    )
    big = Llama(CONFIG).to(dtype)
    accumulated = Llama(CONFIG).to(dtype)
    accumulated.load_state_dict(big.state_dict())

    big_loss = train_step(big, micro_batches(tokens, 32, 8), grad_acc=1)
    acc_loss = train_step(accumulated, micro_batches(tokens, 32, 2), grad_acc=4)

    assert abs(big_loss - acc_loss) < 1e-5
    for (name, a), (_, b) in zip(
        big.named_parameters(), accumulated.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=atol, msg=name)

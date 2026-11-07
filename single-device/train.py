import argparse
import random
import time
from pathlib import Path

import torch

from dtrain.data import micro_batches
from dtrain.memory import num_params
from dtrain.model import Llama, LlamaConfig
from dtrain.train import train_step

CORPUS = Path(__file__).resolve().parents[1] / "LICENSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--mbs", type=int, default=8)
    parser.add_argument("--grad-acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    tokens = torch.tensor(list(CORPUS.read_bytes()), dtype=torch.long)
    config = LlamaConfig(max_position_embeddings=args.seq)
    model = Llama(config, recompute=args.recompute)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = micro_batches(tokens, args.seq, args.mbs)
    tokens_per_step = args.mbs * args.grad_acc * args.seq

    print(f"{num_params(config):,} parameters · {tokens.numel():,} byte tokens")
    print(f"global batch = mbs {args.mbs} × grad_acc {args.grad_acc} × seq {args.seq}")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        loss = train_step(model, batches, args.grad_acc)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            rate = step * tokens_per_step / (time.perf_counter() - start)
            print(f"step {step:>4}  loss {loss:.3f}  {rate:,.0f} tokens/s")


if __name__ == "__main__":
    main()

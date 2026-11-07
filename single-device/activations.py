import torch

from dtrain.model import Llama, LlamaConfig
from dtrain.train import loss_fn


def saved_bytes(model, ids):
    seen = {}

    def pack(t):
        seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = loss_fn(model(ids[:, :-1]), ids[:, 1:])
    loss.backward()
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    return sum(size for ptr, size in seen.items() if ptr not in params)


def main():
    torch.manual_seed(0)
    print("bytes autograd keeps for backward, weights excluded")
    print(f"{'seq':>6}  {'stored':>12}  {'recompute':>12}")
    for seq in (128, 256, 512, 1024):
        config = LlamaConfig(max_position_embeddings=seq, num_hidden_layers=4)
        ids = torch.randint(0, config.vocab_size, (1, seq + 1))
        plain = Llama(config)
        recomputed = Llama(config, recompute=True)
        recomputed.load_state_dict(plain.state_dict())
        a, b = saved_bytes(plain, ids), saved_bytes(recomputed, ids)
        print(f"{seq:>6}  {a / 1e6:>9.2f} MB  {b / 1e6:>9.2f} MB")


if __name__ == "__main__":
    main()

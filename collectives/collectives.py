import torch.distributed as dist


def fmt(t):
    if t is None:
        return "-"
    if isinstance(t, list):
        return "[" + ", ".join(fmt(x) for x in t) + "]"
    return str([int(v) for v in t.tolist()])


def table(title, before, after):
    rows = [None] * dist.get_world_size()
    dist.gather_object((fmt(before), fmt(after)), rows if dist.get_rank() == 0 else None, dst=0)
    if dist.get_rank() == 0:
        print(f"\n{title}")
        width = max(len(b) for b, _ in rows)
        for r, (b, a) in enumerate(rows):
            print(f"  rank {r}  {b:<{width}}  ->  {a}")

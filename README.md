# Learning distributed training

I'm learning how models train across GPUs. This repo is where I work through the questions with code, diagrams, and small experiments.

- [Why distributed training?](why-distributed/README.md): build a model, count its parameters, and work out training memory and compute.
- [How GPUs communicate](collectives/README.md): follow tensors across four processes, then see what breaks when one process skips a call.

## Run the code

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/kuluruvineeth/learning-distributed-training.git
cd learning-distributed-training
git switch collectives
uv sync --locked --python 3.12
uv run pytest -q
uv run torchrun --nproc-per-node=4 --master-port=29500 collectives/collectives.py
```

These examples run on CPU. The [collectives notes](collectives/README.md) include the macOS setup and the intentional deadlock example. The [model notes](why-distributed/README.md) cover the parameter, memory, and compute calculations.

## Diagrams

Download a board and open it in [Excalidraw](https://excalidraw.com):

- [Why distributed training?](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/why-distributed-training/why-distributed-training.excalidraw)
- [How GPUs communicate](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/collectives/collectives.excalidraw)

The commits follow the build. `main` contains the work I've shared so far. Topic branches and tags keep the matching snapshots: [`why-distributed-training`](https://github.com/kuluruvineeth/learning-distributed-training/tree/why-distributed-training) and [`collectives`](https://github.com/kuluruvineeth/learning-distributed-training/tree/collectives).

## License

[Apache-2.0](LICENSE).

# Learning distributed training

I'm learning how models train across GPUs. This repo is where I work through the questions with code, diagrams, and small experiments.

[Why distributed training?](why-distributed/README.md) starts with a model on one device: how its parts fit together, how much memory training needs, and how long the computation takes. The examples use Llama models to make the numbers concrete.

## Run the code

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/kuluruvineeth/learning-distributed-training.git
cd learning-distributed-training
git switch why-distributed-training
uv sync --locked --python 3.12
uv run pytest -q
uv run python why-distributed/params.py
```

The code runs on a CPU. Large-model calculations use PyTorch's meta device to work with shapes without allocating the full weights. The tests compare a small model's outputs and gradients with Transformers and check the parameter and FLOP formulas.

The [topic notes](why-distributed/README.md) contain the other commands and the assumptions behind the estimates.

## Diagrams

[Download the Excalidraw board](https://github.com/kuluruvineeth/learning-distributed-training/releases/download/why-distributed-training/why-distributed-training.excalidraw) and open it in [Excalidraw](https://excalidraw.com).

## Finding your way around

- `src/dtrain/`: the model, loss function, weight mapping, and memory and FLOP formulas.
- `why-distributed/`: experiments and notes.
- `tests/`: checks against the formulas and Transformers.
- `boards/`: diagrams.

The commits follow the build. `main` contains the work I've shared so far; topic branches and tags keep the matching snapshots available. The current snapshot is [`why-distributed-training`](https://github.com/kuluruvineeth/learning-distributed-training/tree/why-distributed-training).

## License

[Apache-2.0](LICENSE).

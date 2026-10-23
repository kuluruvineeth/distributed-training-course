# Why distributed training?

How much memory does training need, and how long would one GPU take? These scripts work through both questions, starting with the model itself.

Llama 3 8B is the main example. The calculations also cover 70B and 405B configurations. Large-model calculations use PyTorch's meta device; the numerical tests use a small model that fits on a laptop.

Run the commands below from the repository root after [setting up the environment](../README.md#run-the-code).

## Follow one sentence

```bash
uv run python why-distributed/walkthrough.py
uv run python why-distributed/walkthrough.py --text "What happens inside attention?"
```

The script traces token IDs through embeddings, attention, the MLP, and the output projection. It prints the shapes and counts the matrix multiplications, then checks the total with PyTorch's FLOP counter.

It downloads the Llama 3 tokenizer from `NousResearch/Meta-Llama-3-8B` on the first run. It does not download model weights.

The implementation is in [`model.py`](../src/dtrain/model.py): RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU, and the decoder stack. [`test_model.py`](../tests/test_model.py) compares logits and gradients with Transformers using the same small configuration and weights.

## Count the parameters

```bash
uv run python why-distributed/params.py
uv run python why-distributed/params.py --model "Llama 3 70B"
```

For the 8B configuration, the count is **8,030,261,248 parameters**. About 70.2% are in the MLP, 16.7% in attention, and 13.1% in the embedding and output projection. RMSNorm adds a small remainder.

The script counts the actual module shapes. The tests check those counts against the formula in [`memory.py`](../src/dtrain/memory.py).

## Turn parameters into memory

```bash
uv run python why-distributed/memory.py
```

The mixed-precision Adam setup used here has:

| State | Bytes per parameter |
|---|---:|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam first and second moments | 8 |
| fp32 gradient accumulation buffer | 4 |
| Total | 20 |

For 8B, that is **160.6 GB of training state**, before activations and framework overhead. This is a particular setup: omitting the separate fp32 accumulation buffer changes the estimate to 16 bytes per parameter.

The script also shows why precision matters. Subtracting `0.001` from a bf16 value of `1.0` rounds back to `1.0`. Keeping a master copy in fp32 allows that update to accumulate. Repeated small gradient additions have a similar rounding problem.

Activation memory depends on the architecture, sequence length, microbatch size, and attention implementation. The activation figures printed by this script are estimates for the stated setup.

## Turn computation into time

```bash
uv run python why-distributed/compute.py
```

For a matrix multiplication `(m × k) @ (k × n)`, the usual count is `2mkn` FLOPs. Applied to a linear layer, that gives roughly two FLOPs per weight per token in forward. Backward adds two matrix multiplications, bringing training to roughly six.

Attention also does work that is not attached to a learned weight matrix. The script calculates that term separately. [`test_flops.py`](../tests/test_flops.py) checks both terms against PyTorch's counter.

The worked GPU-time estimate uses:

- `6ND`, with 8.03 billion parameters and 15 trillion tokens.
- An H100 dense bf16 peak of 989 TFLOP/s.
- 40% model FLOP utilization.

Under those assumptions, the estimate is about **58 years on one GPU**, or **20.6 days on 1,024 GPUs with perfect scaling**. The `6ND` approximation omits the separate attention term. Real distributed runs also have communication and other overheads.

## Compare model sizes

```bash
uv run python why-distributed/estimate.py
```

This prints the memory and compute estimates for 8B, 70B, and 405B with the same assumptions. The minimum GPU counts cover training state only; activations and runtime overhead need additional room.

[Run the checks](../tests)

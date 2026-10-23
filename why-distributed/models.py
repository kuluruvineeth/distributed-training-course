from dtrain.model import LlamaConfig

MODELS = {
    "Llama 3 8B": LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "Llama 3 70B": LlamaConfig(
        vocab_size=128256,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
    ),
    "Llama 3.1 405B": LlamaConfig(
        vocab_size=128256,
        hidden_size=16384,
        intermediate_size=53248,
        num_hidden_layers=126,
        num_attention_heads=128,
        num_key_value_heads=8,
    ),
}

HUGGING_FACE_PARAMS = {
    "Llama 3 8B": 8_030_261_248,
    "Llama 3 70B": 70_553_706_496,
    "Llama 3.1 405B": 405_853_388_800,
}

REPORTED_GPU_HOURS = {"Llama 3 8B": 1.46e6, "Llama 3 70B": 7.0e6, "Llama 3.1 405B": 30.84e6}

TOKENS = 15e12
SEQ_LENGTH = 8192
GPU_MEMORY = 80e9
GPU_PEAK_FLOPS = 989e12
MFU = 0.40

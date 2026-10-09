import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from .models.llama import LlamaForCausalLM


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def load_model(
    device: torch.device,
) -> LlamaForCausalLM:
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    config = hf.config
    if not isinstance(config, LlamaConfig):
        raise TypeError("Expected a LlamaConfig for the TinyLlama checkpoint")

    model = LlamaForCausalLM(config)

    model.load_state_dict(
        hf.state_dict(),
        strict=True,
    )

    del hf

    model = model.to(device)
    model.eval()

    return model

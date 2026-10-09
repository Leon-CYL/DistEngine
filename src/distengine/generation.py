import torch

from .layers.sampler import GreedySampler
from .models.llama import LlamaForCausalLM


@torch.inference_mode()
def generate_greedy(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected input_ids with shape [1, T]")

    if input_ids.shape[1] == 0:
        raise ValueError("The prompt must contain at least one token")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    model.eval()
    sampler = GreedySampler()
    token_ids = input_ids

    for _ in range(max_new_tokens):
        logits = model(token_ids)
        next_token = sampler(logits[:, -1, :])

        token_ids = torch.cat(
            (token_ids, next_token.unsqueeze(-1)),
            dim=-1,
        )

        if next_token.item() == eos_token_id:
            break

    return token_ids
import torch

from .layers.sampler import GreedySampler
from .models.llama import LlamaForCausalLM
from .request import Request
from .paged_kv_cache import PagedKVCache


@torch.inference_mode()
def generate_greedy_cached(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    cache: PagedKVCache,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected input_ids with shape [1, T]")
    if input_ids.shape[1] == 0:
        raise ValueError("The prompt must contain at least one token")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if cache.sequence_length != 0:
        raise ValueError("Generation requires an empty cache")
    if input_ids.shape[1] + max_new_tokens > cache.capacity:
        raise ValueError("Insufficient cache capacity")

    model.eval()
    sampler = GreedySampler()
    request = Request("manual-cached", input_ids[0].tolist(), max_new_tokens, eos_token_id)
    request.start()
    pending_ids = input_ids

    while not request.is_finished:
        logits = model(pending_ids, cache=cache)
        next_token = sampler(logits[:, -1, :])
        request.append_token(int(next_token.item()))
        if not request.is_finished:
            pending_ids = next_token.reshape(1, 1)

    return torch.tensor(
        [request.token_ids], dtype=input_ids.dtype, device=input_ids.device,
    )


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

    request = Request(
        request_id="manual",
        prompt_token_ids=input_ids[0].tolist(),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    request.start()

    while not request.is_finished:
        token_ids = torch.tensor(
            [request.token_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        logits = model(token_ids)
        next_token = sampler(logits[:, -1, :])

        request.append_token(int(next_token.item()))

    return torch.tensor(
        [request.token_ids],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

"""Versioned CPU KV snapshots; physical block IDs never cross workers."""

import hashlib
import json

import torch

from .paged_kv_cache import PagedKVCache
from .request import Request


def model_identity(model) -> str:
    """Identity for immutable inference weights and their model configuration."""
    if not hasattr(model, "_transfer_identity"):
        digest = hashlib.sha256(
            json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()
        )

        pinned = getattr(model, "_checkpoint_identity", None)

        if pinned is not None:
            digest.update(pinned.encode())
        else:
            for name, tensor in model.state_dict().items():
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

        model._transfer_identity = digest.hexdigest()

    return model._transfer_identity


@torch.inference_mode()
def export_prefill(runner, request: Request) -> dict:
    if request.is_finished or request.num_generated_tokens != 1:
        raise ValueError("Only unfinished, freshly prefetched requests can transfer")

    cache = runner.caches[request.request_id]

    if cache.sequence_length != request.num_prompt_tokens:
        raise ValueError("Prefill KV length does not match the prompt")

    layers = [cache.read(layer) for layer in range(cache.num_layers)]

    # clone() also isolates CPU-worker exports; .cpu() alone may alias CPU data.
    with torch.inference_mode(False):
        keys = torch.stack([k for k, _ in layers]).detach().cpu().clone().contiguous()
        values = torch.stack([v for _, v in layers]).detach().cpu().clone().contiguous()

    return {
        "version": 1,
        "model_identity": model_identity(runner.model),
        "request_id": request.request_id,
        "prompt_token_ids": request.prompt_token_ids.copy(),
        "generated_token_ids": request.generated_token_ids.copy(),
        "max_new_tokens": request.max_new_tokens,
        "eos_token_id": request.eos_token_id,
        "cache_length": cache.sequence_length,
        "keys": keys,
        "values": values,
    }


@torch.inference_mode()
def import_prefill(runner, payload: dict) -> Request:
    """Validate before allocation; publish cache only after every layer copies."""
    required = {
        "version",
        "model_identity",
        "request_id",
        "prompt_token_ids",
        "generated_token_ids",
        "max_new_tokens",
        "eos_token_id",
        "cache_length",
        "keys",
        "values",
    }

    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Malformed KV transfer payload")

    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Unsupported KV transfer version")

    if payload["model_identity"] != model_identity(runner.model):
        raise ValueError("KV transfer model/configuration mismatch")

    if (
        not isinstance(payload["prompt_token_ids"], list)
        or not isinstance(payload["generated_token_ids"], list)
        or len(payload["generated_token_ids"]) != 1
        or type(payload["max_new_tokens"]) is not int
        or payload["max_new_tokens"] <= 0
        or type(payload["eos_token_id"]) is not int
    ):
        raise ValueError("Invalid transferred request metadata")

    if (
        not isinstance(payload["request_id"], str)
        or not payload["request_id"]
        or any(
            type(token) is not int
            for token in payload["prompt_token_ids"] + payload["generated_token_ids"]
        )
    ):
        raise ValueError("Invalid request ID or token metadata")

    request = Request(
        payload["request_id"],
        payload["prompt_token_ids"],
        payload["max_new_tokens"],
        payload["eos_token_id"],
    )
    request.start()
    request.append_token(payload["generated_token_ids"][0])
    runner._validate_request(request)

    if request.is_finished:
        raise ValueError("Finished requests must complete at prefill without KV transfer")

    length = payload["cache_length"]

    if type(length) is not int or length != request.num_prompt_tokens:
        raise ValueError("Transferred cache length does not match request history")

    if (
        request.request_id in runner.caches
        or request.request_id in runner.block_manager.request_blocks
    ):
        raise ValueError("Request is already allocated on destination")

    config = runner.model.config
    heads = config.num_key_value_heads or config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    shape = (config.num_hidden_layers, 1, heads, length, head_dim)

    for tensor in (payload["keys"], payload["values"]):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.device.type != "cpu"
            or tensor.dtype != runner.dtype
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError("KV must be finite CPU tensors with matching shape and dtype")

    maximum = request.num_prompt_tokens + request.max_new_tokens - 1

    if maximum > runner.block_manager.num_blocks * runner.block_manager.block_size:
        raise ValueError("Transferred request exceeds destination pool capacity")

    table = runner.block_manager.allocate(request.request_id, length)
    cache = None

    try:
        cache = PagedKVCache(
            config.num_hidden_layers,
            heads,
            head_dim,
            runner.block_manager.num_blocks,
            runner.block_manager.block_size,
            table,
            runner.device,
            runner.dtype,
            storage=runner.storage,
        )

        for layer in range(config.num_hidden_layers):
            cache.append(
                layer,
                payload["keys"][layer].to(runner.device),
                payload["values"][layer].to(runner.device),
            )

        if runner.device.type == "cuda":
            # ACK only after all host-to-device copies are complete.
            torch.cuda.synchronize(runner.device)

        runner.caches[request.request_id] = cache
        return request

    except Exception:
        cache = None
        runner.block_manager.free(request.request_id)
        raise

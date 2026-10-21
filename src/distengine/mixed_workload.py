"""Deterministic prompt selection independent of arrival generation."""

from dataclasses import dataclass, replace
import math
import random

from .workload import Workload, generate_workload


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    length_class: str
    token_ids: tuple[int, ...]
    text: str = ""


def generate_mixed_workload(
    *,
    arrival_mode,
    rate_rps: float,
    num_requests: int,
    arrival_seed: int,
    prompt_seed: int,
    prompt_bank: tuple[Prompt, ...],
    class_weights: dict[str, float],
    max_new_tokens: int,
    eos_token_id: int,
    vocab_size: int,
    max_context_tokens: int,
    kv_capacity_tokens: int,
) -> Workload:
    """Use exact class quotas (largest remainder), then shuffle with prompt_seed.

    Within each class prompts are sampled uniformly with replacement. Class
    names are sorted to make dictionary insertion order irrelevant. Workload.seed
    records the arrival seed; keep selection settings in a companion manifest.
    Context includes all outputs; KV capacity excludes the final pending token.
    """
    if type(prompt_seed) is not int:
        raise ValueError("prompt_seed must be an integer")
    for value in (vocab_size, max_context_tokens, kv_capacity_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError("Vocabulary, context and KV capacities must be positive integers")
    if not class_weights or any(not isinstance(c, str) or not c for c in class_weights):
        raise ValueError("class_weights requires nonempty class names")
    weights = []
    classes = sorted(class_weights)
    for name in classes:
        weight = class_weights[name]
        if type(weight) not in (int, float):
            raise ValueError("Class weights must be finite nonnegative numbers")
        try:
            valid = math.isfinite(weight) and weight >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Class weights must be finite nonnegative numbers")
        weights.append(weight)
    # Normalize before summing to avoid overflowing large finite weights.
    largest = max(weights)
    if largest == 0:
        raise ValueError("At least one class weight must be positive")
    weights = [w / largest for w in weights]
    groups = {name: [] for name in classes}
    seen = set()
    for prompt in prompt_bank:
        if (
            not isinstance(prompt, Prompt)
            or not isinstance(prompt.prompt_id, str)
            or not prompt.prompt_id
        ):
            raise ValueError("Prompt bank requires Prompt records with nonempty IDs")
        if prompt.prompt_id in seen:
            raise ValueError("Duplicate prompt ID")
        if prompt.length_class not in groups:
            raise ValueError("Prompt class is missing from class_weights")
        if not isinstance(prompt.text, str):
            raise ValueError("Prompt text must be a string")
        # Reuse the request generator's validation even for unused bank entries.
        generate_workload(
            arrival_mode=arrival_mode,
            rate_rps=rate_rps,
            num_requests=0,
            seed=arrival_seed,
            prompt_token_ids=prompt.token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        if any(token >= vocab_size for token in prompt.token_ids):
            raise ValueError("Prompt token is outside the model vocabulary")
        if len(prompt.token_ids) + max_new_tokens > max_context_tokens:
            raise ValueError("Prompt and output limit exceed model context")
        if len(prompt.token_ids) + max_new_tokens - 1 > kv_capacity_tokens:
            raise ValueError("Request exceeds per-worker KV capacity")
        groups[prompt.length_class].append(prompt)
        seen.add(prompt.prompt_id)
    if not prompt_bank or any(not groups[c] and class_weights[c] > 0 for c in classes):
        raise ValueError("Every positive-weight class needs a prompt")
    if eos_token_id != -1 and not 0 <= eos_token_id < vocab_size:
        raise ValueError("EOS must be -1 or a vocabulary token")
    base = generate_workload(
        arrival_mode=arrival_mode,
        rate_rps=rate_rps,
        num_requests=num_requests,
        seed=arrival_seed,
        prompt_token_ids=prompt_bank[0].token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    quotas = [num_requests * (w / sum(weights)) for w in weights]
    counts = [math.floor(q) for q in quotas]
    priority = sorted(range(len(classes)), key=lambda i: (-(quotas[i] - counts[i]), classes[i]))
    for index in priority[: num_requests - sum(counts)]:
        counts[index] += 1
    labels = [c for c, count in zip(classes, counts) for _ in range(count)]
    rng = random.Random(prompt_seed)
    rng.shuffle(labels)
    requests = tuple(
        replace(request, prompt_token_ids=tuple(rng.choice(groups[label]).token_ids))
        for request, label in zip(base, labels)
    )
    return Workload(1, arrival_mode, rate_rps, arrival_seed, requests)

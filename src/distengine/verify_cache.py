import torch
from transformers import LlamaConfig

from .paged_kv_cache import PagedKVCache
from .models.llama import LlamaForCausalLM
from .generation import generate_greedy, generate_greedy_cached


@torch.inference_mode()
def main():
    torch.manual_seed(0)

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )

    model = LlamaForCausalLM(config).eval()

    paged = PagedKVCache(
        num_layers=2, num_kv_heads=2, head_dim=4,
        num_physical_blocks=4, block_size=4, block_table=[3, 0],
        device=torch.device("cpu"), dtype=torch.float32,
    )
    check_cache(model, paged, "paged")


@torch.inference_mode()
def check_cache(model, cache, name):
    ids = torch.tensor([[10, 20, 30, 40, 50, 60]])
    full = model(ids)

    # Prefill the first three tokens.
    prefill = model(ids[:, :3], cache=cache)
    torch.testing.assert_close(
        prefill, full[:, :3], atol=1e-4, rtol=1e-4
    )
    assert cache.sequence_length == 3

    # Decode one token at absolute position three.
    decoded = model(ids[:, 3:4], cache=cache)
    torch.testing.assert_close(
        decoded, full[:, 3:4], atol=1e-4, rtol=1e-4
    )
    assert cache.sequence_length == 4

    # Process two more tokens together.
    chunk = model(ids[:, 4:], cache=cache)
    torch.testing.assert_close(
        chunk, full[:, 4:], atol=1e-4, rtol=1e-4
    )
    assert cache.sequence_length == 6

    # Reset and reuse the same storage.
    cache.reset()
    repeated = model(ids, cache=cache)
    torch.testing.assert_close(
        repeated, full, atol=1e-4, rtol=1e-4
    )
    assert cache.sequence_length == 6

    print("PASS: cached prefill, decode, chunk, and reset")

    prompt = ids[:, :3]
    # An unreachable EOS isolates length stopping.
    expected = generate_greedy(model, prompt, 3, -1)
    cache.reset()
    actual = generate_greedy_cached(model, prompt, 3, -1, cache)
    assert torch.equal(actual, expected)
    assert cache.sequence_length == actual.shape[1] - 1

    try:
        generate_greedy_cached(model, prompt, 3, -1, cache)
    except ValueError as error:
        assert str(error) == "Generation requires an empty cache"
    else:
        raise AssertionError("Expected rejection of nonempty cache")

    cache.reset()
    try:
        generate_greedy_cached(model, prompt, 6, -1, cache)
    except ValueError as error:
        assert str(error) == "Insufficient cache capacity"
    else:
        raise AssertionError("Expected rejection of insufficient capacity")
    assert cache.sequence_length == 0
    print("PASS: cached generation, length tracking, and cache guards")

    print(f"PASS: {name} model and generation checks")


if __name__ == "__main__":
    main()

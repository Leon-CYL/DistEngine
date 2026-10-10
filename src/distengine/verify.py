"""Run stage 4/5/6 correctness checks against the pinned HF checkpoint."""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .generation import generate_greedy, generate_greedy_cached
from .paged_kv_cache import PagedKVCache
from .hf_reference import choose_device
from .load_model import MODEL_ID, REVISION, load_model


@torch.inference_mode()
def main() -> None:
    device = choose_device()
    print(f"Checkpoint: {REVISION}; device: {device}; dtype: float32", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    eos = tokenizer.eos_token_id
    assert eos is not None
    prompts = [
        "What is the capital of France?",
        "Explain why the sky is blue in two sentences.",
        "Complete this sentence: The best way to learn programming is",
    ]
    inputs = []
    for prompt in prompts:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs.append(tokenizer(formatted, add_special_tokens=False,
                                return_tensors="pt")["input_ids"].to(device))

    custom = load_model(device)
    custom_logits = []
    custom_tokens = []
    for ids in inputs:
        logits = custom(ids)
        assert logits.shape == (1, ids.shape[1], custom.config.vocab_size)
        assert torch.isfinite(logits).all(), "Nonfinite custom logits"
        custom_logits.append(logits.cpu())
        output = generate_greedy(custom, ids, 32, eos)
        assert torch.equal(output[:, :ids.shape[1]], ids), "Prompt changed"
        generated = output[0, ids.shape[1]:].cpu()
        assert 1 <= len(generated) <= 32
        assert eos not in generated[:-1].tolist(), "Generated past EOS"
        assert len(generated) == 32 or generated[-1].item() == eos
        custom_tokens.append(generated)

        capacity = ids.shape[1] + 32
        block_size = 4
        num_blocks = (capacity + block_size - 1) // block_size
        cache_args = dict(
            num_layers=custom.config.num_hidden_layers,
            num_kv_heads=custom.config.num_key_value_heads,
            head_dim=custom.config.hidden_size // custom.config.num_attention_heads,
            device=ids.device,
            dtype=next(custom.parameters()).dtype,
        )
        caches = [
            ("paged", PagedKVCache(
                num_physical_blocks=num_blocks, block_size=block_size,
                block_table=list(reversed(range(num_blocks))), **cache_args,
            )),
        ]
        for cache_name, cache in caches:
            cached_output = generate_greedy_cached(custom, ids, 32, eos, cache)
            assert torch.equal(cached_output, output), "Cached and uncached generation differ"
            assert cache.sequence_length == cached_output.shape[1] - 1
            print(f"PASS: {cache_name} generation matches uncached", flush=True)

            cache.reset()
            prefill_logits = custom(ids, cache=cache)
            torch.testing.assert_close(
                prefill_logits.cpu(), custom_logits[-1], atol=1e-4, rtol=1e-4,
            )
            suffix_ids = cached_output[:, ids.shape[1]:]
            suffix_logits = custom(suffix_ids, cache=cache)
            full_logits = custom(cached_output)
            torch.testing.assert_close(
                suffix_logits, full_logits[:, ids.shape[1]:, :], atol=1e-4, rtol=1e-4,
            )
            assert cache.sequence_length == cached_output.shape[1]
            print(f"PASS: {cache_name} real-checkpoint cached logits", flush=True)
            del cached_output, prefill_logits, suffix_ids, suffix_logits, full_logits

        del caches, cache

    # Appending a future token must not affect any previous position.
    extended = torch.cat((inputs[0], inputs[0][:, -1:]), dim=1)
    prefix_logits = custom(extended)[:, :-1].cpu()
    torch.testing.assert_close(prefix_logits, custom_logits[0], atol=1e-4, rtol=1e-4)
    print("PASS: shapes, finite logits, prompt preservation, causal masking", flush=True)

    repeated = generate_greedy(custom, inputs[0], 32, eos)[0, inputs[0].shape[1]:].cpu()
    assert torch.equal(repeated, custom_tokens[0]), "Repeat run differed"
    # Disable EOS stopping with an unreachable token ID to isolate the limit.
    limited = generate_greedy(custom, inputs[0], 3, -1)
    assert limited.shape[1] == inputs[0].shape[1] + 3
    print("PASS: repeatability and output-limit stopping", flush=True)

    del custom, logits, output, extended, prefix_logits, repeated, limited
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    for prompt, ids, expected_logits, expected_tokens in zip(
        prompts, inputs, custom_logits, custom_tokens
    ):
        actual = hf(ids, use_cache=False).logits.cpu()
        error = (actual - expected_logits).abs()
        print(f"{prompt}\n  Logit error: max={error.max().item():.6g}, "
              f"mean={error.mean().item():.6g}", flush=True)
        # Initial float32 tolerance: investigate failures rather than loosen blindly.
        torch.testing.assert_close(expected_logits, actual, atol=1e-4, rtol=1e-4)
        output = hf.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
            do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=eos, pad_token_id=eos,
        )
        generated = output[0, ids.shape[1]:].cpu()
        assert torch.equal(expected_tokens, generated), (
            f"Token mismatch for {prompt!r}: custom={expected_tokens.tolist()}, "
            f"HF={generated.tolist()}"
        )
        print(f"  PASS: logits and exact token agreement ({len(generated)} tokens)", flush=True)

    assert custom_tokens[0][-1].item() == eos, "Factual example did not reach EOS"
    print("PASS: factual EOS stopping\nAll stage 4/5/6 checks passed.", flush=True)


if __name__ == "__main__":
    main()

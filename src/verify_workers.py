import torch
from unittest.mock import patch
from transformers import LlamaConfig

from distengine.workers import PrefillWorker, DecodeWorker
from generation import generate_greedy
from models.llama import LlamaForCausalLM
from verify_runner import raises


def check_empty(worker):
    assert not worker.requests
    assert not worker.runner.caches
    assert not worker.runner.block_manager.request_blocks
    assert worker.info()["num_free_blocks"] == worker.info()["num_blocks"]


def check_boundaries():
    # Both role entry points can finish immediately during local prefill.
    for cls in (PrefillWorker, DecodeWorker):
        worker = cls(make_model(), 2, 4)
        start = worker.prefill if cls is PrefillWorker else worker.prepare_local_fixture
        try:
            first = start("one", [10, 20, 30], 1, -1)
            assert first["status"] == "FINISHED" and first["reason"] == "length"
            check_empty(worker)
            eos = first["generated_token_ids"][0]
            stopped = start("eos", [10, 20, 30], 4, eos)
            assert stopped["reason"] == "eos"
            assert len(stopped["generated_token_ids"]) == 1
            check_empty(worker)
            snapshot = start("active", [10], 3, -1)
            snapshot["generated_token_ids"].clear()
            assert worker.requests["active"].num_generated_tokens == 1
            before = worker.info()
            for args in [("active", [20], 2, -1), ("huge", [10] * 9, 1, -1),
                         ("empty", [], 2, -1), ("bad-token", [64], 2, -1),
                         ("bad-limit", [10], 0, -1)]:
                raises(ValueError, lambda: start(*args))
                assert worker.info() == before
            raises(KeyError, lambda: worker.release("missing"))
            worker.close()
            worker.close()
            check_empty(worker)
            raises(RuntimeError, lambda: start("closed", [10], 1, -1))
        finally:
            worker.close()
    print("PASS: immediate EOS, one-token limit, rejection atomicity, snapshots, close")


def check_capacity_retry():
    worker = DecodeWorker(make_model(), 2, 4)
    try:
        # Initialize unused slots too, so the exact storage comparison below
        # never depends on uninitialized values (including possible NaNs).
        worker.runner.storage.k_cache.zero_()
        worker.runner.storage.v_cache.zero_()
        worker.prepare_local_fixture("A", [10, 20, 30, 40], 3, -1)
        worker.prepare_local_fixture("B", [50], 2, -1)
        request = worker.requests["A"]
        before_tokens = request.token_ids
        cache = worker.runner.caches["A"]
        before_length = cache.sequence_length
        before_k = worker.runner.storage.k_cache.clone()
        before_v = worker.runner.storage.v_cache.clone()
        before_tables = {key: value.copy() for key, value in
                         worker.runner.block_manager.request_blocks.items()}
        raises(RuntimeError, lambda: worker.step("A"))
        assert worker.requests["A"] is request
        assert request.token_ids == before_tokens
        assert cache.sequence_length == before_length
        assert worker.runner.block_manager.request_blocks == before_tables
        assert torch.equal(before_k, worker.runner.storage.k_cache)
        assert torch.equal(before_v, worker.runner.storage.v_cache)
        worker.release("B")
        worker.step("A")
        result = worker.step("A")
        expected = generate_greedy(worker.runner.model,
                                  torch.tensor([[10, 20, 30, 40]]), 3, -1)[0].tolist()
        assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
        check_empty(worker)
    finally:
        worker.close()
    print("PASS: decode capacity exhaustion preserves state and succeeds after release")


def check_partial_write_failures():
    for phase in ("prefill", "decode"):
        worker = DecodeWorker(make_model(), 4, 4)
        try:
            worker.prepare_local_fixture("survivor", [50], 2, -1)
            if phase == "decode":
                worker.prepare_local_fixture("failed", [10, 20, 30, 40], 3, -1)

            def fail_forward(input_ids, positions=None, cache=None):
                k = torch.zeros(1, cache.num_kv_heads, input_ids.shape[1],
                                cache.head_dim, device=worker.runner.device,
                                dtype=worker.runner.dtype)
                cache.append(0, k, k)
                raise RuntimeError("Injected failure after layer-zero KV write")

            with patch.object(worker.runner.model, "forward", side_effect=fail_forward):
                operation = (lambda: worker.step("failed")) if phase == "decode" else (
                    lambda: worker.prepare_local_fixture("failed", [10, 20], 3, -1))
                raises(RuntimeError, operation)
            assert list(worker.requests) == ["survivor"]
            assert list(worker.runner.caches) == ["survivor"]
            assert list(worker.runner.block_manager.request_blocks) == ["survivor"]
            result = worker.step("survivor")
            expected = generate_greedy(worker.runner.model,
                                      torch.tensor([[50]]), 2, -1)[0].tolist()
            assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
            check_empty(worker)
            # A new request can use the released blocks after the failure.
            worker.prepare_local_fixture("recovered", [10], 1, -1)
            check_empty(worker)
        finally:
            worker.close()
    print("PASS: partial-write prefill/decode failure cleanup and surviving-request recovery")


def make_model():
    torch.manual_seed(0)

    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval()


def main():
    torch.set_num_threads(1)

    prompt = [10, 20, 30]
    limit = 4
    eos = -1  # Disable EOS stopping for this test.

    reference_model = make_model()
    expected = generate_greedy(
        reference_model,
        torch.tensor([prompt]),
        limit,
        eos,
    )[0].tolist()

    prefill = PrefillWorker(make_model(), num_blocks=8, block_size=4)
    decode = DecodeWorker(make_model(), num_blocks=8, block_size=4)

    try:
        first = prefill.prefill("prefill-example", prompt, limit, eos)

        assert first["generated_token_ids"] == [
            expected[len(prompt)]
        ]
        assert prefill.info()["active_request_ids"] == ["prefill-example"]

        prefill.release("prefill-example")
        assert prefill.info()["num_free_blocks"] == 8

        result = decode.prepare_local_fixture(
            "decode-example", prompt, limit, eos
        )

        while result["status"] != "FINISHED":
            result = decode.step("decode-example")

        actual = (
            result["prompt_token_ids"]
            + result["generated_token_ids"]
        )

        assert actual == expected
        assert result["reason"] == "length"
        assert decode.info()["active_request_ids"] == []
        assert decode.info()["num_free_blocks"] == 8

        print("PASS: prefill, decode, baseline agreement, block release")
    finally:
        prefill.close()
        decode.close()

    check_boundaries()
    check_capacity_retry()
    check_partial_write_failures()
    print("All ordinary worker checks passed.")


if __name__ == "__main__":
    main()

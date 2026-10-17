"""CPU checks for admission primitives and the serial pull coordinator."""

from unittest.mock import patch

import torch

from distengine.holding_pool import PrefillHoldingPool
from distengine.admission import pull_one
from distengine.paged_kv_cache import PagedKVCache
from distengine.workers import PrefillWorker, DecodeWorker
from generation import generate_greedy
from verify_runner import raises
from verify_workers import make_model, check_empty


def check_metadata():
    source = PrefillWorker(make_model(), 8, 4, PrefillHoldingPool(2))
    try:
        assert source.peek_ready() is None
        source.prefill("a", [10, 20], 4, -1)
        source.prefill("b", [30], 3, -1)
        expected = {"request_id": "a", "num_prompt_tokens": 2, "max_new_tokens": 4}
        before = source.info()
        with patch.object(source, "export_request", side_effect=AssertionError("No export")):
            descriptor = source.peek_ready()
            assert descriptor == expected
            descriptor["request_id"] = "changed"
            assert source.peek_ready() == expected
        assert source.info() == before
        assert source.prefill_pool.request_ids() == ["a", "b"]
        source.release("a")
        assert source.peek_ready()["request_id"] == "b"
        source.close()
        assert len(source.prefill_pool) == 0
        raises(RuntimeError, source.peek_ready)
    finally:
        source.close()
    print("PASS: empty/FIFO metadata, snapshot isolation, no export, release and closed guard")


def check_capacity():
    target = DecodeWorker(make_model(), 3, 4)
    try:
        before = target.info()
        for invalid in (0, -1, True, 1.5, None):
            raises(ValueError, lambda: target.can_admit(invalid, 4))
            raises(ValueError, lambda: target.can_admit(5, invalid))
        raises(ValueError, lambda: target.can_admit(10, 4))
        assert target.can_admit(5, 6) is True
        assert target.can_admit(5, 8) is True  # Exactly twelve cached tokens.
        assert target.info() == before
        assert not target.runner.block_manager.request_blocks
        target.prepare_local_fixture("occupied", [10], 2, -1)
        before = target.info()
        table = target.runner.block_manager.get_block_table("occupied")
        assert target.can_admit(5, 6) is False  # Three blocks required, two free.
        assert target.can_admit(5, 4) is True  # Exactly two blocks required.
        assert target.info() == before
        assert target.runner.block_manager.get_block_table("occupied") == table
        target.release("occupied")
        assert target.can_admit(5, 6) is True
        target.close()
        raises(RuntimeError, lambda: target.can_admit(5, 6))
    finally:
        target.close()
    print("PASS: capacity validation, rounding, permanent/temporary rejection and read-only queries")


def check_reservation():
    source = PrefillWorker(make_model(), 8, 4, PrefillHoldingPool(2))
    target = DecodeWorker(make_model(), 3, 4)
    prompt = [10, 20, 30, 40, 50]
    try:
        source.prefill("a", prompt, 6, -1)
        payload = source.export_request("a")
        for invalid in (0, 1, None, "yes"):
            raises(ValueError, lambda: target.import_request(payload, reserve_full=invalid))
            check_empty(target)

        # A prior successful query does not reserve memory: another request
        # can consume capacity before import. Allocation remains authoritative.
        assert target.can_admit(5, 6)
        target.prepare_local_fixture("occupied", [10], 2, -1)
        before = target.info()
        raises(RuntimeError, lambda: target.import_request(payload, reserve_full=True))
        assert target.info() == before
        assert list(target.runner.caches) == ["occupied"]
        assert list(target.runner.block_manager.request_blocks) == ["occupied"]
        assert source.peek_ready()["request_id"] == "a"
        target.release("occupied")

        append = PagedKVCache.append

        def fail_second(cache, layer, k, v):
            if layer == 1:
                raise RuntimeError("Injected reserved import failure")
            return append(cache, layer, k, v)

        with patch.object(PagedKVCache, "append", fail_second):
            raises(RuntimeError, lambda: target.import_request(payload, reserve_full=True))
        check_empty(target)
        assert source.peek_ready()["request_id"] == "a"

        with patch.object(target.runner.model, "forward", side_effect=AssertionError("No recompute")):
            target.import_request(payload, reserve_full=True)
        manager = target.runner.block_manager
        table = manager.get_block_table("a")
        cache = target.runner.caches["a"]
        assert len(table) == 3 and manager.num_free_blocks == 0
        assert cache.capacity == 12 and cache.sequence_length == 5
        for layer in range(cache.num_layers):
            k, v = cache.read(layer)
            assert torch.equal(k, payload["keys"][layer])
            assert torch.equal(v, payload["values"][layer])
        source.release("a")
        check_empty(source)
        while "a" in target.requests:
            assert manager.get_block_table("a") == table
            assert manager.num_free_blocks == 0
            result = target.step("a")
        expected = generate_greedy(target.runner.model, torch.tensor([prompt]), 6, -1)[0].tolist()
        assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
        check_empty(target)

        # Existing callers retain prompt-only allocation by default.
        target.import_request(payload)
        assert len(manager.get_block_table("a")) == 2
        assert target.runner.caches["a"].sequence_length == 5
        target.release("a")
        check_empty(target)
    finally:
        source.close()
        target.close()
    print("PASS: reservations, stale-query rejection, full rollback, KV equality, fixed blocks and baseline")


def check_pull():
    source = PrefillWorker(make_model(), 8, 4, PrefillHoldingPool(2))
    target = DecodeWorker(make_model(), 3, 4)
    prompts = {"a": [10, 20, 30, 40, 50], "b": [30]}
    limits = {"a": 6, "b": 2}
    try:
        assert pull_one(source, target) is None
        check_empty(target)
        for rid in prompts:
            source.prefill(rid, prompts[rid], limits[rid], -1)
        target.prepare_local_fixture("occupied", [10], 2, -1)
        before = source.info(), target.info()
        snapshot = source.export_request("a")
        with patch.object(source, "export_request", side_effect=AssertionError("Denied pull exported KV")):
            assert pull_one(source, target) is None
        assert (source.info(), target.info()) == before
        assert source.prefill_pool.request_ids() == ["a", "b"]
        for layer in range(source.runner.caches["a"].num_layers):
            k, v = source.runner.caches["a"].read(layer)
            assert torch.equal(k, snapshot["keys"][layer])
            assert torch.equal(v, snapshot["values"][layer])
        target.release("occupied")

        append = PagedKVCache.append

        def fail_second(cache, layer, k, v):
            if layer == 1:
                raise RuntimeError("Injected pull copy failure")
            return append(cache, layer, k, v)

        with patch.object(PagedKVCache, "append", fail_second):
            raises(RuntimeError, lambda: pull_one(source, target))
        check_empty(target)
        assert source.prefill_pool.request_ids() == ["a", "b"]

        for rid in prompts:
            result = pull_one(source, target)
            assert result["request_id"] == rid
            assert rid not in source.requests and rid not in source.runner.caches
            table = target.runner.block_manager.get_block_table(rid)
            while rid in target.requests:
                assert target.runner.block_manager.get_block_table(rid) == table
                result = target.step(rid)
            expected = generate_greedy(target.runner.model,
                                      torch.tensor([prompts[rid]]), limits[rid], -1)[0].tolist()
            assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
            check_empty(target)
        check_empty(source)
        assert pull_one(source, target) is None

        # Permanent rejection must preserve source ownership too.
        source.prefill("oversized", [10] * 10, 4, -1)
        with patch.object(source, "export_request", side_effect=AssertionError("Oversized pull exported KV")):
            raises(ValueError, lambda: pull_one(source, target))
        assert source.peek_ready()["request_id"] == "oversized"
        check_empty(target)
        source.release("oversized")

        # Once import succeeds, release failure must surface without deleting
        # the destination or automatically issuing a second import.
        source.prefill("release-failed", [10], 2, -1)
        with patch.object(source, "release", side_effect=RuntimeError("Release failure")):
            raises(RuntimeError, lambda: pull_one(source, target))
        assert "release-failed" in source.requests
        assert "release-failed" in target.requests
    finally:
        source.close()
        target.close()
    check_empty(source)
    check_empty(target)
    print("PASS: pull FIFO, denial without export, retry, rollback, continuation and release failure")


def main():
    torch.set_num_threads(1)
    check_metadata()
    check_capacity()
    check_reservation()
    check_pull()
    print("All admission checks passed.")


if __name__ == "__main__":
    main()

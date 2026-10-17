"""CPU checks for bounded holding and prefill-worker ownership."""

from unittest.mock import patch

import torch

from distengine.holding_pool import PrefillHoldingPool
from distengine.workers import PrefillWorker, DecodeWorker
from generation import generate_greedy
from verify_runner import raises
from verify_workers import make_model, check_empty


def check_pool():
    for capacity in (0, -1, True, 1.5):
        raises(ValueError, lambda: PrefillHoldingPool(capacity))
    pool = PrefillHoldingPool(2)
    assert len(pool) == 0 and pool.peek() is None and not pool.is_full()
    for rid in ("", None, 123):
        raises(ValueError, lambda: pool.add(rid))
    pool.add("a")
    pool.add("b")
    assert pool.is_full() and pool.peek() == "a"
    raises(ValueError, lambda: pool.add("a"))
    raises(RuntimeError, lambda: pool.add("c"))
    raises(KeyError, lambda: pool.remove("missing"))
    assert pool.request_ids() == ["a", "b"]
    ids = pool.request_ids()
    ids.clear()
    assert pool.request_ids() == ["a", "b"]
    pool.remove("b")
    pool.add("c")
    pool.remove("a")
    assert pool.peek() == "c"
    pool.add("a")
    assert pool.request_ids() == ["c", "a"]
    pool.remove("c")
    pool.remove("a")
    assert len(pool) == 0 and pool.peek() is None
    print("PASS: pool validation, FIFO, bounds, rejection atomicity, copies and reuse")


def check_worker():
    pool = PrefillHoldingPool(2)
    source = PrefillWorker(make_model(), 8, 4, pool)
    target = DecodeWorker(make_model(), 8, 4)
    try:
        first = source.prefill("finished", [10], 1, -1)
        check_empty(source)
        eos = first["generated_token_ids"][0]
        assert source.prefill("eos", [10], 4, eos)["reason"] == "eos"
        check_empty(source)
        source.prefill("a", [10, 20], 4, -1)
        before = source.info()
        raises(ValueError, lambda: source.prefill("invalid", [64], 4, -1))

        def fail_forward(input_ids, positions=None, cache=None):
            k = torch.zeros(1, cache.num_kv_heads, input_ids.shape[1],
                            cache.head_dim)
            cache.append(0, k, k)
            raise RuntimeError("Injected partial prefill failure")

        with patch.object(source.runner.model, "forward", side_effect=fail_forward):
            raises(RuntimeError, lambda: source.prefill("failed", [30], 4, -1))
        assert source.info() == before
        assert pool.request_ids() == ["a"]
        assert list(source.runner.caches) == ["a"]
        assert list(source.runner.block_manager.request_blocks) == ["a"]

        source.prefill("b", [30], 4, -1)
        before = source.info()
        with patch.object(source.runner.model, "forward",
                          side_effect=AssertionError("Full pool must reject before forward")):
            raises(RuntimeError, lambda: source.prefill("c", [40], 4, -1))
        assert source.info() == before and pool.request_ids() == ["a", "b"]

        payload = source.export_request(pool.peek())
        assert pool.request_ids() == ["a", "b"]
        raises(ValueError, lambda: target.import_request({**payload, "version": 2}))
        assert source.info() == before and pool.request_ids() == ["a", "b"]
        with patch.object(target.runner.model, "forward",
                          side_effect=AssertionError("Import must not recompute prompt")):
            target.import_request(payload)
        source.release("a")
        assert pool.request_ids() == ["b"]
        while "a" in target.requests:
            result = target.step("a")
        expected = generate_greedy(target.runner.model,
                                   torch.tensor([[10, 20]]), 4, -1)[0].tolist()
        assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
        check_empty(target)
        source.prefill("a", [40], 4, -1)
        assert pool.request_ids() == ["b", "a"]
        source.close()
        source.close()
        check_empty(source)
        raises(RuntimeError, lambda: source.prefill("closed", [10], 4, -1))
        check_empty(source)
    finally:
        source.close()
        target.close()
    print("PASS: completion, partial failure, bounded admission, handoff retry, continuation and cleanup")


def main():
    torch.set_num_threads(1)
    check_pool()
    check_worker()
    print("All holding pool checks passed.")


if __name__ == "__main__":
    main()

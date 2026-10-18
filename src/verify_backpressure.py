"""CPU coordinator scheduling, backpressure, and completion recovery checks."""

from unittest.mock import patch

import torch

from distengine.coordinator import Coordinator
from distengine.holding_pool import PrefillHoldingPool
from distengine.workers import PrefillWorker, DecodeWorker
from generation import generate_greedy
from verify_runner import raises
from verify_workers import make_model, check_empty


def workers(target_blocks=2):
    return (PrefillWorker(make_model(), 4, 4, PrefillHoldingPool(1)),
            DecodeWorker(make_model(), target_blocks, 4))


def check_pressure():
    source, target = workers()
    try:
        c = Coordinator(source, target)
        for rid in ("a", "b", "c"):
            c.submit(rid, [10, 20, 30], 4, -1)
        assert c.step() == []  # A prefills.
        assert c.step() == []  # A transfers; B prefills.
        with patch.object(source, "prefill", side_effect=AssertionError("Must wait upstream")):
            assert c.step() == []  # A decodes; B cannot transfer; C waits.
        assert [s.request_id for s in c.waiting] == ["c"]
        assert source.prefill_pool.request_ids() == ["b"]
        assert list(c.decode_ready) == ["a"]
        results = []
        for _ in range(30):
            if not c.has_pending():
                break
            results.extend(c.step())
        assert not c.has_pending()
        assert [r["request_id"] for r in results] == ["a", "b", "c"]
        expected = generate_greedy(target.runner.model, torch.tensor([[10, 20, 30]]), 4, -1)[0].tolist()
        assert all(r["prompt_token_ids"] + r["generated_token_ids"] == expected for r in results)
        assert c.drain_completed() == [] and c.step() == []
        check_empty(source)
        check_empty(target)
    finally:
        source.close()
        target.close()
    print("PASS: upstream waiting, decode progress, automatic resumption, exact outputs and cleanup")


def check_completion_recovery():
    source, target = workers()
    try:
        c = Coordinator(source, target)
        c.submit("a", [10], 2, -1)
        c.step()
        c.step()
        with patch.object(c, "_pull_one", side_effect=RuntimeError("Later tick failure")):
            raises(RuntimeError, c.step)
        assert not c.decode_ready and not c.has_pending()
        results = c.drain_completed()
        assert [r["request_id"] for r in results] == ["a"]
        assert c.drain_completed() == []
        c.submit("immediate", [10], 1, -1)
        assert c.step()[0]["request_id"] == "immediate"
        check_empty(source)
        check_empty(target)
        source.prefill("external", [10], 2, -1)
        raises(ValueError, lambda: Coordinator(source, target))
    finally:
        source.close()
        target.close()
    print("PASS: completion retention on later failure, exactly-once delivery, immediate finish and fresh-worker guard")


def check_round_robin():
    source, target = workers(target_blocks=4)
    try:
        c = Coordinator(source, target)
        c.submit("a", [10], 5, -1)
        c.submit("b", [20], 5, -1)
        for _ in range(3):
            c.step()
        assert list(c.decode_ready) == ["a", "b"]
        c.step()
        assert list(c.decode_ready) == ["b", "a"]
        c.step()
        assert list(c.decode_ready) == ["a", "b"]
        for _ in range(20):
            if not c.has_pending():
                break
            c.step()
        assert not c.has_pending()
        check_empty(source)
        check_empty(target)
    finally:
        source.close()
        target.close()
    print("PASS: round-robin rotation without duplicate or finished queue entries")


def main():
    torch.set_num_threads(1)
    check_pressure()
    check_completion_recovery()
    check_round_robin()
    print("All CPU backpressure checks passed.")


if __name__ == "__main__":
    main()

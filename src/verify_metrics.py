"""Driver timing arithmetic and metrics-enabled CPU inference checks."""

from contextlib import ExitStack
from dataclasses import replace
import json
from unittest.mock import patch

import torch

from distengine.metric import LatencyLimits, MetricsRecorder, percentile
from distengine.multi_decode_coordinator import MultiDecodeCoordinator
from distengine.multi_prefill_coordinator import MultiPrefillCoordinator
from distengine.workload import Workload, replay_workload
from verify_multi_decode import workers, drain
from verify_runner import raises
from verify_workers import check_empty
from verify_workload import FakeClock, with_offsets


def check_arithmetic():
    clock = FakeClock()
    m = MetricsRecorder(clock=clock, started_at=100)
    m.submitted("a", 0)
    clock.now += 1
    m.prefill_dispatched("a")
    clock.now += 2
    m.observe("a", generated_tokens=1, finished=False)
    clock.now += 1
    m.observe("a", generated_tokens=2, finished=False)
    clock.now += 2
    m.observe("a", generated_tokens=3, finished=True)
    clock.now += 1
    m.submitted("one", 6)
    m.prefill_dispatched("one")
    clock.now += 1
    m.observe("one", generated_tokens=1, finished=True)
    m.submitted("bad", 8)
    m.failed("bad", RuntimeError("rejected"))
    m.submitted("unfinished", 8)
    m.orchestration_failed(RuntimeError("recoverable"))
    m.finish()
    report = m.report(LatencyLimits(ttft_s=3, tpot_s=1, e2e_s=10))
    a, one, bad, unfinished = report["requests"]
    assert a["prefill_wait_s"] == 1 and a["ttft_s"] == 3
    assert a["e2e_s"] == 6 and a["tpot_s"] == 1.5
    assert one["submission_delay_s"] == 1 and one["tpot_s"] is None
    assert report["completed"] == 2 and report["failed"] == 1 and report["unfinished"] == 1
    assert report["duration_s"] == 8
    assert report["request_throughput_rps"] == .25
    assert report["completed_output_tokens_per_s"] == .5
    assert report["goodput_rps"] == .125
    assert report["latencies"]["tpot_s"]["count"] == 1
    assert len(report["orchestration_errors"]) == 1
    clock.now += 100
    assert m.report(LatencyLimits(3, 1, 10)) == report
    assert percentile([], .5) is None
    assert percentile([10, 0], .95) == 9.5
    json.dumps(report, allow_nan=False)
    raises(ValueError, lambda: m.submitted("late", 0))
    print("PASS: exact TTFT/TPOT/E2E, wait/delay, goodput, failures and frozen report window")


def check_invalid_events():
    clock = FakeClock()
    m = MetricsRecorder(clock)
    raises(ValueError, lambda: m.submitted("early", 1))
    m.submitted("a", 0)
    raises(ValueError, lambda: m.submitted("a", 0))
    raises(ValueError, lambda: m.observe("a", generated_tokens=1, finished=False))
    m.prefill_dispatched("a")
    raises(ValueError, lambda: m.prefill_dispatched("a"))
    raises(ValueError, lambda: m.observe("a", generated_tokens=2, finished=False))
    m.observe("a", generated_tokens=1, finished=False)
    raises(ValueError, lambda: m.observe("a", generated_tokens=1, finished=False))
    m.observe("a", generated_tokens=2, finished=True)
    raises(ValueError, lambda: m.observe("a", generated_tokens=3, finished=True))
    raises(ValueError, lambda: m.failed("a", RuntimeError()))
    print("PASS: duplicate/out-of-order token events and invalid lifecycle rejection")


def run_cpu(enabled, single_target=False):
    sources, targets = workers()
    clock = FakeClock()
    m = MetricsRecorder(clock) if enabled else None
    try:
        if single_target:
            c = MultiPrefillCoordinator(sources, targets[0], metrics=m)
        else:
            c = MultiDecodeCoordinator(sources, targets, metrics=m)
        workload = with_offsets([0, .25, .25, 1, 2, 2])
        workload = replace(workload, requests=tuple(
            replace(r, max_new_tokens=1 if i == 0 else 4)
            for i, r in enumerate(workload.requests)
        ))
        with ExitStack() as stack:
            for worker in (*sources, *targets):
                method = "prefill" if worker in sources else "step"
                original = getattr(worker, method)
                def timed(*args, original=original, **kwargs):
                    clock.now += .125
                    return original(*args, **kwargs)
                stack.enter_context(patch.object(worker, method, side_effect=timed))
            results = replay_workload(workload, c, clock=clock, sleep=clock.sleep, metrics=m)
        if m is not None:
            report = m.report(LatencyLimits(10, 10, 10))
            assert report["completed"] == len(results) == 6
            assert not report["failed"] and not report["unfinished"]
            for result in results:
                timing = m.requests[result["request_id"]]
                assert len(timing.token_observed_s) == len(result["generated_token_ids"])
                assert timing.prefill_dispatched_s >= timing.submitted_s
                assert timing.token_observed_s[0] >= timing.prefill_dispatched_s
                assert timing.finished_s == timing.token_observed_s[-1]
            assert report["requests"][0]["tpot_s"] is None
            assert "0" not in getattr(c, "decode_assignments", {})
        for w in (*sources, *targets):
            check_empty(w)
        return {r["request_id"]: r for r in results}
    finally:
        for w in (*sources, *targets):
            w.close()


def check_replay_guards():
    sources, targets = workers()
    try:
        clock = FakeClock()
        m = MetricsRecorder(clock)
        c = MultiDecodeCoordinator(sources, targets, metrics=m)
        raises(ValueError, lambda: replay_workload(with_offsets([0]), c, metrics=MetricsRecorder(clock)))
        raises(ValueError, lambda: replay_workload(with_offsets([0]), c, clock=lambda: 0, metrics=m))
        assert not c.seen_ids
        oversized = replace(with_offsets([0]), requests=(
            replace(with_offsets([0]).requests[0], prompt_token_ids=tuple([10] * 100)),
        ))
        raises(ValueError, lambda: replay_workload(oversized, c, clock=clock, sleep=clock.sleep, metrics=m))
        assert m.requests["0"].error is not None and not c.seen_ids
    finally:
        for w in (*sources, *targets):
            w.close()
    print("PASS: mismatched recorders/clocks and rejected-submission failure metrics")


def check_failure_classification():
    sources, targets = workers()
    clock = FakeClock()
    m = MetricsRecorder(clock)
    try:
        c = MultiDecodeCoordinator(sources, targets, metrics=m)
        c.submit("a", [10], 4, -1)
        c.submit("b", [20], 4, -1)
        c.step()
        with patch.object(sources[0], "release", side_effect=RuntimeError("release failure")):
            raises(RuntimeError, c.step)
        assert all(t.error is None for t in m.requests.values())
        assert len(m.requests["a"].token_observed_s) == 1
        assert len(drain(c)) == 2
        assert len(m.requests["a"].token_observed_s) == 4
        c.submit("retired", [10], 4, -1)
        c.step()
        c._pull_targets()
        target = targets[c.decode_assignments["retired"]]
        def fail(request_id):
            target.release(request_id)
            raise RuntimeError("retired decode")
        with patch.object(target, "step", side_effect=fail):
            raises(RuntimeError, c.step)
        assert m.requests["retired"].error is not None
        assert m.report(LatencyLimits(10, 10, 10))["failed"] == 1
        for w in (*sources, *targets):
            check_empty(w)
    finally:
        for w in (*sources, *targets):
            w.close()
    print("PASS: recoverable release failures stay live; retired decode failures are recorded")


def check_local_observation_order():
    sources, targets = workers()
    clock = FakeClock()
    metrics = MetricsRecorder(clock)
    try:
        c = MultiDecodeCoordinator(sources, targets, metrics=metrics)
        c.submit("a", [10], 2, -1)
        c.submit("b", [20], 2, -1)
        c.step()
        c._pull_targets()
        ids = [q[0] for q in c.decode_ready]
        with ExitStack() as stack:
            for index, target in enumerate(targets):
                original = target.step
                def timed(request_id, original=original, delay=index + 1):
                    clock.now += delay
                    return original(request_id)
                stack.enter_context(patch.object(target, "step", side_effect=timed))
            c._decode_targets()
        assert metrics.requests[ids[0]].finished_s == 1
        assert metrics.requests[ids[1]].finished_s == 3
    finally:
        for worker in (*sources, *targets):
            worker.close()
    print("PASS: local result observed before executing a slower subsequent decoder")


def main():
    torch.set_num_threads(1)
    check_arithmetic()
    check_invalid_events()
    assert run_cpu(False) == run_cpu(True)
    assert run_cpu(True, single_target=True) == run_cpu(True)
    print("PASS: metrics on/off exact CPU outputs, single/multi decode, first tokens and handoff counted once")
    check_replay_guards()
    check_failure_classification()
    check_local_observation_order()


if __name__ == "__main__":
    main()

"""CPU checks for reproducible workload files and scheduled coordinator replay."""

from collections import deque
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch

from distengine.holding_pool import PrefillHoldingPool
from distengine.multi_prefill_coordinator import MultiPrefillCoordinator
from distengine.workers import DecodeWorker, PrefillWorker
from distengine.workload import (
    ScheduledRequest, Workload, generate_workload, load_workload,
    replay_workload, save_workload,
)
from generation import generate_greedy
from verify_runner import raises
from verify_workers import check_empty, make_model


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now = max(self.now + seconds, math.nextafter(self.now, math.inf))


class FakeCoordinator:
    def __init__(self, clock, step_duration=0.0):
        self.clock = clock
        self.step_duration = step_duration
        self.pending = deque()
        self.events = []

    def submit(self, request_id, prompt_token_ids, max_new_tokens, eos_token_id):
        assert isinstance(prompt_token_ids, list)
        self.events.append(("submit", request_id, self.clock()))
        self.pending.append(request_id)

    def has_pending(self):
        return bool(self.pending)

    def step(self):
        self.events.append(("step", self.clock()))
        self.clock.now += self.step_duration
        return [{"request_id": self.pending.popleft()}]


def make_workload(mode="constant", count=6):
    requests = generate_workload(
        arrival_mode=mode, rate_rps=4.0, num_requests=count, seed=42,
        prompt_token_ids=[10, 20, 30], max_new_tokens=4, eos_token_id=-1,
    )
    return Workload(1, mode, 4.0, 42, tuple(requests))


def with_offsets(offsets):
    return Workload(1, "constant", 1.0, 42, tuple(
        ScheduledRequest(str(i), offset, (10,), 4, -1)
        for i, offset in enumerate(offsets)
    ))


def check_generation():
    constant = make_workload()
    assert [r.arrival_offset_s for r in constant.requests] == [0, .25, .5, .75, 1, 1.25]
    state = random.getstate()
    poisson = make_workload("poisson")
    assert random.getstate() == state
    assert poisson == make_workload("poisson")
    rng = random.Random(42)
    offset = 0.0
    for request in poisson.requests:
        offset += rng.expovariate(4.0)
        assert request.arrival_offset_s == offset
    assert not make_workload(count=0).requests
    print("PASS: constant arrivals, seeded Poisson gaps and isolated randomness")


def check_files():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "workload.json"
        for workload in (make_workload(), make_workload("poisson"), make_workload(count=0)):
            save_workload(workload, path)
            assert load_workload(path) == workload
        for text in ("{", "null", "[]", "{}"):
            path.write_text(text, encoding="utf-8")
            raises(ValueError, lambda: load_workload(path))
        for mutate in (
            lambda p: p.update(schema_version=2),
            lambda p: p.update(rate_rps=float("inf")),
            lambda p: p["requests"][1].update(request_id=p["requests"][0]["request_id"]),
            lambda p: p["requests"][0].update(arrival_offset_s=-1),
            lambda p: p["requests"][1].update(arrival_offset_s=-.1),
            lambda p: p["requests"][0].update(arrival_offset_s=float("nan")),
            lambda p: p["requests"][0].update(prompt_token_ids=[True]),
        ):
            payload = asdict(make_workload())
            mutate(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            raises(ValueError, lambda: load_workload(path))
        save_workload(make_workload(), path)
        before = path.read_text()
        raises(ValueError, lambda: save_workload(replace(make_workload(), rate_rps=-1), path))
        assert path.read_text() == before
    print("PASS: JSON round trips, invalid files and validation before overwrite")


def check_replay():
    for offsets, duration in (([], 0), ([0, 2, 5], 0), ([0, 1, 2, 2], 3)):
        clock = FakeClock()
        coordinator = FakeCoordinator(clock, duration)
        results = replay_workload(with_offsets(offsets), coordinator, clock=clock, sleep=clock.sleep)
        expected_ids = [str(i) for i in range(len(offsets))]
        assert [r["request_id"] for r in results] == expected_ids
        submissions = [e for e in coordinator.events if e[0] == "submit"]
        assert [e[1] for e in submissions] == expected_ids
        assert all(e[2] - 100 >= offset for e, offset in zip(submissions, offsets))
        assert not coordinator.has_pending()
        if offsets == [0, 2, 5]:
            assert clock.sleeps == [2, 3]
        if duration:
            assert [e[0] for e in coordinator.events[:5]] == ["submit", "step", "submit", "submit", "submit"]
            assert not clock.sleeps
    clock = FakeClock()
    coordinator = FakeCoordinator(clock)
    raises(ValueError, lambda: replay_workload(with_offsets([2, 1]), coordinator))
    assert not coordinator.events
    print("PASS: empty replay, idle gaps, tied/overdue arrivals and final draining")


def check_cpu_integration(mode):
    workers = []
    try:
        for _ in range(2):
            workers.append(PrefillWorker(make_model(), 4, 4, PrefillHoldingPool(1)))
        target = DecodeWorker(make_model(), 2, 4)
        workers.append(target)
        coordinator = MultiPrefillCoordinator(workers[:2], target)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            original = make_workload(mode)
            save_workload(original, path)
            workload = load_workload(path)
            assert workload == original
            clock = FakeClock()
            results = replay_workload(workload, coordinator, clock=clock, sleep=clock.sleep)
        assert len(results) == len(workload.requests)
        by_id = {r["request_id"]: r for r in results}
        assert set(by_id) == {r.request_id for r in workload.requests}
        for request in workload.requests:
            expected = generate_greedy(
                target.runner.model, torch.tensor([request.prompt_token_ids]),
                request.max_new_tokens, request.eos_token_id,
            )[0].tolist()
            result = by_id[request.request_id]
            assert result["status"] == "FINISHED"
            assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
        assert not coordinator.has_pending()
        assert not any(coordinator.waiting) and not coordinator.decode_ready
        assert coordinator.drain_completed() == []
        for worker in workers:
            check_empty(worker)
    finally:
        for worker in workers:
            worker.close()
    print(f"PASS: {mode} save/load/replay, six exact CPU outputs, drained queues and released KV")


def main():
    torch.set_num_threads(1)
    check_generation()
    check_files()
    check_replay()
    for mode in ("constant", "poisson"):
        check_cpu_integration(mode)


if __name__ == "__main__":
    main()

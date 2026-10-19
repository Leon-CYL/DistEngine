"""CPU multi-decode ownership, backpressure and failure reconciliation checks."""

from unittest.mock import patch

import torch

from distengine.holding_pool import PrefillHoldingPool
from distengine.multi_decode_coordinator import MultiDecodeCoordinator
from distengine.workers import PrefillWorker, DecodeWorker
from generation import generate_greedy
from verify_runner import raises
from verify_workers import make_model, check_empty


def workers():
    sources = [PrefillWorker(make_model(), 4, 4, PrefillHoldingPool(1)) for _ in range(2)]
    targets = [DecodeWorker(make_model(), 3, 4) for _ in range(2)]
    return sources, targets


def drain(c, limit=150):
    results = []
    for _ in range(limit):
        if not c.has_pending():
            break
        results.extend(c.step())
    assert not c.has_pending(), "Multi-decode scheduler failed to drain"
    return results


def check_outputs():
    sources, targets = workers()
    try:
        raises(ValueError, lambda: MultiDecodeCoordinator(sources, []))
        raises(ValueError, lambda: MultiDecodeCoordinator([], targets))
        raises(ValueError, lambda: MultiDecodeCoordinator(sources, [targets[0], targets[0]]))
        raises(ValueError, lambda: MultiDecodeCoordinator([sources[0], sources[0]], targets))
        c = MultiDecodeCoordinator(sources, targets)
        prompts = [[10, 20, 30], [40], [10, 20, 30, 40], [50, 60], [20], [30]]
        for i, prompt in enumerate(prompts):
            c.submit(str(i), prompt, 8, -1)
        raises(ValueError, lambda: c.submit("0", [10], 8, -1))
        assert c.step() == []
        assert c.step() == []
        assert all(c.decode_ready)
        assert set(c.decode_assignments.values()) == {0, 1}
        # Both targets reserve all blocks; held sources must remain unchanged.
        before = [s.info() for s in sources]
        assert c.step() == []
        assert [s.info() for s in sources] == before
        owners = dict(c.decode_assignments)
        results = drain(c)
        assert all(c.decode_assignments[rid] == owner for rid, owner in owners.items())
        assert len(results) == len(prompts)
        by_id = {r["request_id"]: r for r in results}
        assert len(by_id) == len(prompts)
        for i, prompt in enumerate(prompts):
            expected = generate_greedy(targets[0].runner.model, torch.tensor([prompt]), 8, -1)[0].tolist()
            result = by_id[str(i)]
            assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
            assert result["status"] == "FINISHED"
        assert not any(c.decode_ready) and not c.pending_releases
        assert len(c.decode_assignments) == len(prompts)
        assert c.step() == [] and c.drain_completed() == []
        # Prefill-only completions have no decode assignment.
        c.submit("immediate", [10], 1, -1)
        assert c.step()[0]["request_id"] == "immediate"
        assert "immediate" not in c.decode_assignments
        raises(ValueError, lambda: c.submit("0", [10], 1, -1))
        for w in (*sources, *targets):
            check_empty(w)
    finally:
        for w in (*sources, *targets):
            w.close()
    print("PASS: guards, both decoders, sticky ownership, backpressure, exact outputs and cleanup")


def check_capacity_and_failures():
    sources, targets = workers()
    try:
        c = MultiDecodeCoordinator(sources, targets)
        c.submit("a", [10], 4, -1)
        c.submit("b", [20], 4, -1)
        c.step()
        # A full/blocked target cannot prevent a healthy peer's pull.
        with patch.object(targets[1], "can_admit", return_value=False):
            c._pull_targets()
        assert len(c.decode_assignments) == 1 and set(c.decode_assignments.values()) == {0}
        results = drain(c)
        assert {r["request_id"] for r in results} == {"a", "b"}
        c.submit("release-retry", [10], 4, -1)
        c.step()
        # Import acknowledgement must be tracked even when source release fails.
        source = sources[0]
        with patch.object(source, "release", side_effect=RuntimeError("release failure")):
            raises(RuntimeError, c._pull_targets)
        assert "release-retry" in c.decode_assignments
        owner = c.decode_assignments["release-retry"]
        assert list(c.decode_ready[owner]) == ["release-retry"]
        assert c.pending_releases == {"release-retry": 0}
        assert source.peek_ready()["request_id"] == "release-retry"
        assert [r["request_id"] for r in drain(c)] == ["release-retry"]
        assert not c.pending_releases
        # Import failures preserve source state and never assign a destination.
        c.submit("import-retry", [10], 4, -1)
        c.step()
        with patch.object(targets[0], "import_request", side_effect=RuntimeError("import failure")), \
             patch.object(targets[1], "import_request", side_effect=RuntimeError("import failure")):
            raises(RuntimeError, c._pull_targets)
        assert "import-retry" not in c.decode_assignments
        assert any(s.peek_ready() is not None for s in sources)
        assert [r["request_id"] for r in drain(c)] == ["import-retry"]
        for w in (*sources, *targets):
            check_empty(w)
    finally:
        for w in (*sources, *targets):
            w.close()
    print("PASS: full-target bypass, import retry and acknowledged transfer tracking after release failure")


def check_decode_failure():
    sources, targets = workers()
    try:
        c = MultiDecodeCoordinator(sources, targets)
        c.submit("a", [10], 2, -1)
        c.submit("b", [20], 2, -1)
        c.step()
        c._pull_targets()
        failed = c.decode_ready[0][0]
        successful = c.decode_ready[1][0]
        # Simulate a worker failure that retires its own request.
        def fail(request_id):
            targets[0].release(request_id)
            raise RuntimeError("decode failure")
        with patch.object(targets[0], "step", side_effect=fail):
            raises(RuntimeError, c.step)
        assert not any(c.decode_ready)
        assert [r["request_id"] for r in c.drain_completed()] == [successful]
        assert c.decode_assignments[failed] == 0
        assert c.step() == []
        for w in (*sources, *targets):
            check_empty(w)
        # Closing unfinished workers releases held and decoding KV.
        for i in range(6):
            c.submit(f"unfinished-{i}", [10], 8, -1)
        c.step(); c.step()
        assert all(c.decode_ready)
    finally:
        for w in (*sources, *targets):
            w.close()
            check_empty(w)
    print("PASS: decode peer completion survives failure, fixed failed ownership and unfinished cleanup")


def check_all_target_validation():
    sources, targets = workers()
    targets[1].close()
    targets[1] = DecodeWorker(make_model(), 1, 4)
    try:
        c = MultiDecodeCoordinator(sources, targets)
        raises(ValueError, lambda: c.submit("oversized", [10, 20, 30], 4, -1))
        assert not c.seen_ids and c.next_submit == 0 and not any(c.waiting)
        targets[1].close()
        raises(RuntimeError, lambda: c.submit("closed", [10], 1, -1))
    finally:
        for w in (*sources, *targets):
            w.close()
    print("PASS: submission validates every target before mutating queues")


def check_decode_rotation_and_replay():
    from distengine.workload import Workload, generate_workload, replay_workload
    from verify_workload import FakeClock

    sources, targets = workers()
    try:
        c = MultiDecodeCoordinator(sources, targets)
        for i in range(6):
            c.submit(str(i), [10], 4, -1)
        for _ in range(3):
            c._prefill_all()
            c._pull_targets()
        before = [list(q) for q in c.decode_ready]
        assert all(len(q) == 3 for q in before)
        c._decode_targets()
        assert [list(q) for q in c.decode_ready] == [q[1:] + q[:1] for q in before]
        assert len(drain(c)) == 6
        workload = Workload(1, "poisson", 4, 42, tuple(generate_workload(
            arrival_mode="poisson", rate_rps=4, num_requests=6, seed=42,
            prompt_token_ids=[10], max_new_tokens=4, eos_token_id=-1,
        )))
        # The same coordinator can accept new IDs; fixture IDs don't overlap.
        clock = FakeClock()
        results = replay_workload(workload, c, clock=clock, sleep=clock.sleep)
        assert len(results) == 6 and len({r["request_id"] for r in results}) == 6
        for w in (*sources, *targets):
            check_empty(w)
    finally:
        for w in (*sources, *targets):
            w.close()
    print("PASS: per-target round-robin queues and workload replay through multiple decoders")



def check_deferred_ray():
    from distengine.multi_decode_ray_coordinator import MultiDecodeRayCoordinator
    from distengine.metric import MetricsRecorder
    from verify_workload import FakeClock

    events = []

    class Ref:
        def __init__(self, actor, method, args, kwargs):
            self.actor, self.method, self.args, self.kwargs = actor, method, args, kwargs
            self.done = False
            self.error = None

        def get(self):
            if not self.done:
                self.done = True
                events.append(("resolve", self.actor.name, self.method))
                try:
                    args = [a.get() if isinstance(a, Ref) else a for a in self.args]
                    if self.method == "ready":
                        self.result = self.actor.worker.info() | {"actor_id": self.actor.name}
                    else:
                        if self.method == "step":
                            metrics.clock.now += 1
                        self.result = getattr(self.actor.worker, self.method)(*args, **self.kwargs)
                except Exception as exc:
                    self.error = exc
            if self.error is not None:
                raise self.error
            return self.result

    class Method:
        def __init__(self, actor, name):
            self.actor, self.name = actor, name

        def remote(self, *args, **kwargs):
            events.append(("dispatch", self.actor.name, self.name))
            return Ref(self.actor, self.name, args, kwargs)

    class Actor:
        def __init__(self, worker, name):
            self.worker, self.name = worker, name

        def __getattr__(self, name):
            return Method(self, name)

    def get(ref):
        return [get(r) for r in ref] if isinstance(ref, list) else ref.get()

    sources, targets = workers()
    actors = [Actor(w, str(i)) for i, w in enumerate([*sources, *targets])]
    try:
        with patch("ray.get", side_effect=get), patch(
            "ray.wait", side_effect=lambda refs, **kwargs: ([refs[-1]], refs[:-1]),
        ):
            metrics = MetricsRecorder(FakeClock())
            c = MultiDecodeRayCoordinator(actors[:2], actors[2:], metrics=metrics)
            c.submit("a", [10], 2, -1)
            c.submit("b", [20], 2, -1)
            c.step()
            c._pull_targets()
            peer = c.decode_ready[1][0]
            def fail(request_id):
                targets[0].release(request_id)
                raise RuntimeError("remote decode failure")
            events.clear()
            with patch.object(targets[0], "step", side_effect=fail):
                raises(RuntimeError, c.step)
            step_events = [e for e in events if e[2] == "step"]
            assert [e[0] for e in step_events] == ["dispatch", "dispatch", "resolve", "resolve"]
            assert [e[1] for e in step_events[2:]] == ["3", "2"]
            assert [r["request_id"] for r in c.drain_completed()] == [peer]
            assert len(metrics.requests[peer].token_observed_s) == 2
            assert metrics.requests[peer].error is None
            assert metrics.requests[peer].finished_s == 1
            failed = next(t for t in metrics.requests.values() if t.request_id != peer)
            assert failed.error is not None and len(failed.token_observed_s) == 1
            assert failed.finished_s == 2
            assert not c.has_pending()
            for w in (*sources, *targets):
                check_empty(w)
    finally:
        for w in (*sources, *targets):
            w.close()
    print("PASS: deferred Ray dispatch before waiting and reconciliation of successful decode peers")

def main():
    torch.set_num_threads(1)
    check_outputs()
    check_capacity_and_failures()
    check_decode_failure()
    check_all_target_validation()
    check_decode_rotation_and_replay()
    check_deferred_ray()


if __name__ == "__main__":
    main()

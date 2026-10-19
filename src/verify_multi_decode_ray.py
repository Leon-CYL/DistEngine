"""Real Ray CPU checks for two prefill and two decode actors."""

import os
import sys

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
import torch

from distengine.multi_decode_ray_coordinator import MultiDecodeRayCoordinator
from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from generation import generate_greedy
from verify_runner import raises


def exercise(sources, targets, inputs, limit, eos, baselines):
    raises(ValueError, lambda: MultiDecodeRayCoordinator(sources, []))
    raises(ValueError, lambda: MultiDecodeRayCoordinator(sources, [targets[0], targets[0]]))
    raises(ValueError, lambda: MultiDecodeRayCoordinator([sources[0], sources[0]], targets))
    c = MultiDecodeRayCoordinator(sources, targets)
    expected = {}
    for i in range(8):
        case = i % len(inputs)
        c.submit(str(i), inputs[case], limit, eos)
        expected[str(i)] = baselines[case]
    raises(ValueError, lambda: c.submit("0", inputs[0], limit, eos))
    results = []
    placement_history = {}
    ticks = 0
    for _ in range(8 * limit + 30):
        if not c.has_pending():
            break
        results.extend(c.step())
        ticks += 1
        infos = ray.get([t.ready.remote() for t in targets], timeout=30)
        for index, info in enumerate(infos):
            assert set(info["active_request_ids"]) == set(c.decode_ready[index])
            for request_id in info["active_request_ids"]:
                assert c.decode_assignments[request_id] == index
                assert placement_history.setdefault(request_id, index) == index
        if ticks == 2:
            assert all(c.decode_ready), "Both targets should admit work"
        if ticks == 3:
            # All inputs need over half the pool: both targets remain occupied.
            held = ray.get([s.peek_ready.remote() for s in sources], timeout=30)
            assert all(item is not None for item in held)
            assert any(c.waiting)
    assert not c.has_pending(), "Multi-decode Ray scheduler failed to drain"
    assert len(results) == len(expected)
    assert len({r["request_id"] for r in results}) == len(expected)
    assert set(c.decode_assignments.values()) == set(range(len(targets)))
    for result in results:
        assert result["status"] == "FINISHED"
        assert result["prompt_token_ids"] + result["generated_token_ids"] == expected[result["request_id"]]
    assert not any(c.decode_ready) and not c.pending_releases
    assert c.drain_completed() == [] and c.step() == []
    for info in ray.get([w.ready.remote() for w in [*sources, *targets]], timeout=30):
        assert not info["active_request_ids"] and info["num_free_blocks"] == info["num_blocks"]
    print("PASS: both decode actors, fixed ownership on every tick, held/upstream pressure, exact outputs and cleanup", flush=True)
    assignments = dict(c.decode_assignments)
    # Leave held/upstream/decode work and verify explicit caller cleanup.
    for i in range(8):
        c.submit(f"unfinished-{i}", inputs[i % len(inputs)], limit, -1)
    c.step(); c.step()
    assert all(c.decode_ready)
    ray.get([w.close.remote() for w in [*sources, *targets]], timeout=60)
    for info in ray.get([w.ready.remote() for w in [*sources, *targets]], timeout=30):
        assert info["closed"] and not info["active_request_ids"]
        assert info["num_free_blocks"] == info["num_blocks"]
    print("PASS: four actors close and release unfinished held and decode KV", flush=True)
    return {"ticks": ticks, "decode_assignments": assignments, "results": results}


def main():
    ray.init(address="local", num_cpus=4, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors = []
    try:
        for _ in range(2):
            actors.append(PrefillActor.remote(num_blocks=4, block_size=4, max_held_requests=1))
        sources = actors.copy()
        for _ in range(2):
            actors.append(DecodeActor.remote(num_blocks=3, block_size=4))
        targets = actors[2:]
        infos = ray.get([a.ready.remote() for a in actors], timeout=120)
        assert len({i["pid"] for i in infos}) == 4
        assert all(i["pid"] != os.getpid() and i["python_executable"] == sys.executable for i in infos)
        model = build_test_model(0)
        inputs = [[10, 20, 30], [40], [10, 20, 30, 40]]
        baselines = [generate_greedy(model, torch.tensor([ids]), 8, -1)[0].tolist() for ids in inputs]
        exercise(sources, targets, inputs, 8, -1, baselines)
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            finally:
                ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()

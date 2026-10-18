"""Real Ray CPU checks for two prefill actors feeding one decode actor."""

import os
import sys

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray
import torch

from distengine.multi_prefill_ray_coordinator import MultiPrefillRayCoordinator
from distengine.ray_workers import PrefillActor, DecodeActor, build_test_model
from generation import generate_greedy
from verify_runner import raises


def check_workers(sources, target):
    raises(ValueError, lambda: MultiPrefillRayCoordinator([sources[0], sources[0]], target))
    c = MultiPrefillRayCoordinator(sources, target)
    prompts = [[10, 20, 30], [40], [10, 20, 30, 40], [50, 60], [20], [30]]
    for index, prompt in enumerate(prompts):
        c.submit(str(index), prompt, 4, -1)
    assert [[s.request_id for s in q] for q in c.waiting] == [["0", "2", "4"], ["1", "3", "5"]]
    raises(ValueError, lambda: c.submit("0", [10], 4, -1))
    assert c.next_submit == 0
    assert c.step() == []
    held = ray.get([s.peek_ready.remote() for s in sources])
    assert [d["request_id"] for d in held] == ["0", "1"]
    assert c.step() == []
    # The first empty pull rotated the initial priority to source 1.
    assert list(c.decode_ready) == ["1"]
    held = ray.get([s.peek_ready.remote() for s in sources])
    assert [d["request_id"] for d in held] == ["0", "3"]
    assert [[s.request_id for s in q] for q in c.waiting] == [["2", "4"], ["5"]]
    results = []
    for _ in range(100):
        if not c.has_pending():
            break
        results.extend(c.step())
    assert not c.has_pending(), "Multi-prefill scheduler did not drain"
    assert len(results) == len(prompts)
    by_id = {r["request_id"]: r for r in results}
    assert set(by_id) == {str(i) for i in range(len(prompts))}
    model = build_test_model(0)
    for index, prompt in enumerate(prompts):
        expected = generate_greedy(model, torch.tensor([prompt]), 4, -1)[0].tolist()
        result = by_id[str(index)]
        assert result["prompt_token_ids"] + result["generated_token_ids"] == expected
    assert c.drain_completed() == [] and c.step() == []
    for info in ray.get([a.ready.remote() for a in [*sources, target]]):
        assert not info["active_request_ids"]
        assert info["num_free_blocks"] == info["num_blocks"]
    print("PASS: multi-prefill placement, duplicate guards, independent holding, admission, outputs and cleanup")


def main():
    ray.init(address="local", num_cpus=3, include_dashboard=False,
             runtime_env={"py_executable": sys.executable})
    actors = []
    try:
        for _ in range(2):
            actors.append(PrefillActor.remote(num_blocks=4, max_held_requests=1))
        target = DecodeActor.remote(num_blocks=2)
        actors.append(target)
        infos = ray.get([a.ready.remote() for a in actors], timeout=60)
        assert len({info["pid"] for info in infos}) == 3
        check_workers(actors[:2], target)
    finally:
        for actor in actors:
            try:
                ray.get(actor.close.remote(), timeout=10)
            except Exception as exc:
                print(f"Cleanup warning: {exc}")
            finally:
                ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
